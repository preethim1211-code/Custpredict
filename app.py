"""
app.py  –  CustPredict v3
--------------------------
Key change: Self-registration system.
  • Anyone can create their own account (Register page)
  • Every user only sees THEIR OWN customers
  • No shared data between accounts
  • First registered user automatically becomes admin
  • Admin can view all users + audit log
  • All other features (XGBoost, SHAP, PDF, CSV, etc.) unchanged
"""

import os, io, csv, json, re
from datetime import datetime, date
from functools import wraps

import joblib
import numpy as np
import shap

from flask import (Flask, render_template, request, redirect,
                   url_for, flash, jsonify, Response, abort)
from flask_sqlalchemy import SQLAlchemy
from flask_login import (LoginManager, UserMixin, login_user,
                         logout_user, login_required, current_user)
from flask_wtf.csrf import CSRFProtect
from werkzeug.security import generate_password_hash, check_password_hash
import xgboost as xgb

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                 Table, TableStyle, HRFlowable)
from reportlab.lib.units import cm

# ── App setup ──────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.config['SECRET_KEY']                     = 'cpps-v3-secret-2024'
app.config['SQLALCHEMY_DATABASE_URI']        = 'sqlite:///customers.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['WTF_CSRF_ENABLED']               = True
app.config['MAX_CONTENT_LENGTH']             = 5 * 1024 * 1024

db            = SQLAlchemy(app)
csrf          = CSRFProtect(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

# ── Load ML models once ────────────────────────────────────────────────────────
M = 'models'
def load(n): return joblib.load(os.path.join(M, n))

lr_churn       = load('lr_churn.pkl')
rf_churn       = load('rf_churn.pkl')
xgb_churn      = load('xgb_churn.pkl')
lr_value       = load('lr_value.pkl')
rf_value       = load('rf_value.pkl')
xgb_value      = load('xgb_value.pkl')
scaler_churn   = load('scaler_churn.pkl')
scaler_value   = load('scaler_value.pkl')
scaler_kmeans  = load('scaler_kmeans.pkl')
kmeans         = load('kmeans.pkl')
le_gender      = load('le_gender.pkl')
le_value       = load('le_value.pkl')
shap_explainer = load('shap_explainer.pkl')

with open(os.path.join(M, 'meta.json')) as f:
    META = json.load(f)

SEGMENT_NAMES = {0: 'Bronze', 1: 'Silver', 2: 'Gold'}
FEATURE_NAMES = META['feature_names']

# ── Database models ────────────────────────────────────────────────────────────
class User(UserMixin, db.Model):
    """
    Each person registers their own account.
    First user to register becomes 'admin'; all others are 'user'.
    """
    id            = db.Column(db.Integer, primary_key=True)
    username      = db.Column(db.String(80),  unique=True, nullable=False)
    email         = db.Column(db.String(120), unique=True, nullable=False)
    full_name     = db.Column(db.String(120), default='')
    password_hash = db.Column(db.String(200), nullable=False)
    role          = db.Column(db.String(20),  default='user')   # 'admin' | 'user'
    created_at    = db.Column(db.DateTime,    default=datetime.utcnow)

    # Relationship: a user owns many customers
    customers     = db.relationship('Customer', backref='owner', lazy=True,
                                    cascade='all, delete-orphan')

    def set_password(self, pw):   self.password_hash = generate_password_hash(pw)
    def check_password(self, pw): return check_password_hash(self.password_hash, pw)


class Customer(db.Model):
    """
    Belongs to exactly one user (user_id foreign key).
    All queries are filtered by current_user.id so users never see each other's data.
    """
    id                 = db.Column(db.Integer, primary_key=True)
    user_id            = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    name               = db.Column(db.String(120), nullable=False)
    age                = db.Column(db.Integer,  nullable=False)
    gender             = db.Column(db.String(10), nullable=False)
    purchase_freq      = db.Column(db.Integer,  nullable=False)
    total_spending     = db.Column(db.Float,    nullable=False)
    last_purchase_date = db.Column(db.Date,     nullable=False)
    status             = db.Column(db.String(20), default='Active')
    notes              = db.Column(db.Text,     default='')
    # Predictions
    churn_lr           = db.Column(db.String(5))
    churn_rf           = db.Column(db.String(5))
    churn_xgb          = db.Column(db.String(5))
    churn_prob_lr      = db.Column(db.Float, default=0)
    churn_prob_rf      = db.Column(db.Float, default=0)
    churn_prob_xgb     = db.Column(db.Float, default=0)
    value_lr           = db.Column(db.String(10))
    value_rf           = db.Column(db.String(10))
    value_xgb          = db.Column(db.String(10))
    segment            = db.Column(db.String(10))
    shap_json          = db.Column(db.Text, default='{}')
    created_at         = db.Column(db.DateTime, default=datetime.utcnow)


class AuditLog(db.Model):
    id        = db.Column(db.Integer, primary_key=True)
    user      = db.Column(db.String(80))
    action    = db.Column(db.String(200))
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)


@login_manager.user_loader
def load_user(uid):
    return db.session.get(User, int(uid))


# ── Helpers ────────────────────────────────────────────────────────────────────
def admin_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not current_user.is_authenticated or current_user.role != 'admin':
            abort(403)
        return f(*args, **kwargs)
    return decorated


def log_action(action):
    db.session.add(AuditLog(user=current_user.username, action=action))
    db.session.commit()


def my_customers():
    """Base query — always filtered to the logged-in user's data."""
    return Customer.query.filter_by(user_id=current_user.id)


def validate_registration(username, email, password, confirm):
    """Returns list of error strings, empty = valid."""
    errors = []
    if len(username) < 3:
        errors.append('Username must be at least 3 characters.')
    if not re.match(r'^[A-Za-z0-9_]+$', username):
        errors.append('Username may only contain letters, numbers and underscores.')
    if User.query.filter_by(username=username).first():
        errors.append('That username is already taken.')
    if not re.match(r'^[^@]+@[^@]+\.[^@]+$', email):
        errors.append('Please enter a valid email address.')
    if User.query.filter_by(email=email).first():
        errors.append('An account with that email already exists.')
    if len(password) < 6:
        errors.append('Password must be at least 6 characters.')
    if password != confirm:
        errors.append('Passwords do not match.')
    return errors


# ── ML helpers ─────────────────────────────────────────────────────────────────
def build_features(age, gender, purchase_freq, total_spending, last_purchase_date):
    gender_enc      = le_gender.transform([gender])[0]
    days_since_last = (date.today() - last_purchase_date).days
    return np.array([[age, gender_enc, purchase_freq, total_spending, days_since_last]])


def predict_all(X_raw):
    X_churn = scaler_churn.transform(X_raw)
    X_val   = scaler_value.transform(X_raw)
    X_km    = scaler_kmeans.transform(X_raw)

    c_lr  = 'Yes' if lr_churn.predict(X_churn)[0]  == 1 else 'No'
    c_rf  = 'Yes' if rf_churn.predict(X_raw)[0]    == 1 else 'No'
    c_xgb = 'Yes' if xgb_churn.predict(X_raw)[0]   == 1 else 'No'

    p_lr  = round(float(lr_churn.predict_proba(X_churn)[0][1])  * 100, 1)
    p_rf  = round(float(rf_churn.predict_proba(X_raw)[0][1])    * 100, 1)
    p_xgb = round(float(xgb_churn.predict_proba(X_raw)[0][1])   * 100, 1)

    v_lr  = le_value.inverse_transform(lr_value.predict(X_val))[0]
    v_rf  = le_value.inverse_transform(rf_value.predict(X_raw))[0]
    v_xgb = le_value.inverse_transform(xgb_value.predict(X_raw))[0]

    seg_id  = kmeans.predict(X_km)[0]
    segment = SEGMENT_NAMES.get(int(seg_id), 'Unknown')

    sv = shap_explainer.shap_values(X_raw)
    if isinstance(sv, list):
        sv_churn = np.array(sv[1]).flatten()
    else:
        sv_arr = np.array(sv)
        if sv_arr.ndim == 3:
            sv_churn = sv_arr[0, :, 1]
        elif sv_arr.ndim == 2:
            sv_churn = sv_arr[0]
        else:
            sv_churn = sv_arr.flatten()

    shap_dict = {FEATURE_NAMES[i]: round(float(sv_churn[i]), 4)
                 for i in range(len(FEATURE_NAMES))}

    return {
        'churn_lr': c_lr,   'churn_prob_lr':  p_lr,
        'churn_rf': c_rf,   'churn_prob_rf':  p_rf,
        'churn_xgb': c_xgb, 'churn_prob_xgb': p_xgb,
        'value_lr': v_lr,   'value_rf': v_rf, 'value_xgb': v_xgb,
        'segment':  segment, 'shap': shap_dict,
    }


# ════════════════════════════════════════════════════════════════════════════════
#  AUTH ROUTES
# ════════════════════════════════════════════════════════════════════════════════

@app.route('/')
def index():
    return redirect(url_for('dashboard') if current_user.is_authenticated else url_for('register'))


# ── Register ───────────────────────────────────────────────────────────────────
@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))

    if request.method == 'POST':
        username  = request.form.get('username', '').strip()
        email     = request.form.get('email', '').strip().lower()
        full_name = request.form.get('full_name', '').strip()
        password  = request.form.get('password', '')
        confirm   = request.form.get('confirm_password', '')

        errors = validate_registration(username, email, password, confirm)
        if errors:
            for e in errors:
                flash(e, 'danger')
            # Re-render with filled values
            return render_template('register.html',
                                   username=username, email=email, full_name=full_name)

        # First user ever → admin; everyone else → user
        role = 'admin' if User.query.count() == 0 else 'user'

        u = User(username=username, email=email,
                 full_name=full_name, role=role)
        u.set_password(password)
        db.session.add(u)
        db.session.commit()

        login_user(u)
        db.session.add(AuditLog(user=username, action='Registered and logged in'))
        db.session.commit()

        flash(f'Welcome, {username}! Your account has been created.', 'success')
        return redirect(url_for('dashboard'))

    return render_template('register.html', username='', email='', full_name='')


# ── Login ──────────────────────────────────────────────────────────────────────
@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))

    if request.method == 'POST':
        identifier = request.form.get('identifier', '').strip()  # username or email
        password   = request.form.get('password', '')

        # Allow login with either username or email
        u = (User.query.filter_by(username=identifier).first() or
             User.query.filter_by(email=identifier).first())

        if u and u.check_password(password):
            login_user(u)
            db.session.add(AuditLog(user=u.username, action='Logged in'))
            db.session.commit()
            return redirect(url_for('dashboard'))

        flash('Invalid username/email or password.', 'danger')

    return render_template('login.html')


# ── Logout ─────────────────────────────────────────────────────────────────────
@app.route('/logout')
@login_required
def logout():
    log_action('Logged out')
    logout_user()
    return redirect(url_for('login'))


# ════════════════════════════════════════════════════════════════════════════════
#  DASHBOARD  — only current user's data
# ════════════════════════════════════════════════════════════════════════════════
@app.route('/dashboard')
@login_required
def dashboard():
    customers = my_customers().order_by(Customer.created_at.desc()).all()
    total     = len(customers)

    churn_yes = sum(1 for c in customers if c.churn_rf == 'Yes')
    churn_no  = total - churn_yes

    value_counts  = {'High': 0, 'Medium': 0, 'Low': 0}
    seg_counts    = {'Gold': 0, 'Silver': 0, 'Bronze': 0}
    status_counts = {'Active': 0, 'At Risk': 0, 'Churned': 0, 'Retained': 0}

    for c in customers:
        if c.value_rf in value_counts:  value_counts[c.value_rf]  += 1
        if c.segment  in seg_counts:    seg_counts[c.segment]     += 1
        if c.status   in status_counts: status_counts[c.status]   += 1

    from collections import defaultdict
    monthly = defaultdict(int)
    for c in customers:
        monthly[c.created_at.strftime('%b %Y')] += 1
    month_labels = list(monthly.keys())[-6:]
    month_data   = [monthly[k] for k in month_labels]

    avg_spend = round(sum(c.total_spending for c in customers) / total, 2) if total else 0

    stats = {'total': total, 'churn_yes': churn_yes,
             'avg_spend': avg_spend, 'high_value': value_counts['High']}
    chart = {
        'churn':          [churn_yes, churn_no],
        'value':          list(value_counts.values()),
        'segment':        list(seg_counts.values()),
        'status':         list(status_counts.values()),
        'monthly_labels': month_labels,
        'monthly_data':   month_data,
    }
    return render_template('dashboard.html',
                           customers=customers[:10],
                           stats=stats,
                           chart_data=json.dumps(chart))


# ════════════════════════════════════════════════════════════════════════════════
#  PREDICT
# ════════════════════════════════════════════════════════════════════════════════
@app.route('/predict', methods=['GET', 'POST'])
@login_required
def predict():
    result = None
    if request.method == 'POST':
        try:
            name     = request.form['name'].strip()
            age      = int(request.form['age'])
            gender   = request.form['gender']
            pfreq    = int(request.form['purchase_freq'])
            spending = float(request.form['total_spending'])
            lpdate   = datetime.strptime(request.form['last_purchase_date'], '%Y-%m-%d').date()
            notes    = request.form.get('notes', '').strip()

            X_raw  = build_features(age, gender, pfreq, spending, lpdate)
            result = predict_all(X_raw)
            result['name'] = name

            status = 'At Risk' if result['churn_rf'] == 'Yes' else 'Active'

            c = Customer(
                user_id=current_user.id,        # ← tied to this user
                name=name, age=age, gender=gender,
                purchase_freq=pfreq, total_spending=spending,
                last_purchase_date=lpdate, status=status, notes=notes,
                churn_lr=result['churn_lr'],   churn_prob_lr=result['churn_prob_lr'],
                churn_rf=result['churn_rf'],   churn_prob_rf=result['churn_prob_rf'],
                churn_xgb=result['churn_xgb'], churn_prob_xgb=result['churn_prob_xgb'],
                value_lr=result['value_lr'],   value_rf=result['value_rf'],
                value_xgb=result['value_xgb'],
                segment=result['segment'],
                shap_json=json.dumps(result['shap']),
            )
            db.session.add(c)
            db.session.commit()
            result['id'] = c.id

            if result['churn_prob_rf'] > 70:
                flash(f"⚠️ High churn risk for {name} ({result['churn_prob_rf']}%)!", 'warning')

            log_action(f"Added customer: {name} (ID {c.id})")

        except Exception as e:
            flash(f'Prediction error: {e}', 'danger')

    return render_template('predict.html', result=result)


# ════════════════════════════════════════════════════════════════════════════════
#  CUSTOMERS — only this user's records
# ════════════════════════════════════════════════════════════════════════════════
@app.route('/customers')
@login_required
def customers():
    q       = request.args.get('q', '').strip()
    churn_f = request.args.get('churn', '')
    value_f = request.args.get('value', '')
    seg_f   = request.args.get('segment', '')
    sort    = request.args.get('sort', 'date')
    page    = request.args.get('page', 1, type=int)

    query = my_customers()
    if q:       query = query.filter(Customer.name.ilike(f'%{q}%'))
    if churn_f: query = query.filter(Customer.churn_rf == churn_f)
    if value_f: query = query.filter(Customer.value_rf == value_f)
    if seg_f:   query = query.filter(Customer.segment  == seg_f)

    if sort == 'spend': query = query.order_by(Customer.total_spending.desc())
    elif sort == 'age': query = query.order_by(Customer.age)
    else:               query = query.order_by(Customer.created_at.desc())

    total = query.count()
    items = query.offset((page - 1) * 20).limit(20).all()
    pages = max(1, (total + 19) // 20)

    return render_template('customers.html',
                           customers=items, page=page, pages=pages, total=total,
                           q=q, churn_f=churn_f, value_f=value_f,
                           seg_f=seg_f, sort=sort)


# ── Customer detail ─────────────────────────────────────────────────────────────
@app.route('/customer/<int:cid>')
@login_required
def customer_detail(cid):
    # get_or_404 PLUS ownership check
    c = my_customers().filter_by(id=cid).first_or_404()
    shap_data = json.loads(c.shap_json or '{}')
    return render_template('customer_detail.html', c=c, shap_data=shap_data)


# ── Edit ───────────────────────────────────────────────────────────────────────
@app.route('/customer/<int:cid>/edit', methods=['GET', 'POST'])
@login_required
def customer_edit(cid):
    c = my_customers().filter_by(id=cid).first_or_404()
    if request.method == 'POST':
        try:
            c.name               = request.form['name'].strip()
            c.age                = int(request.form['age'])
            c.gender             = request.form['gender']
            c.purchase_freq      = int(request.form['purchase_freq'])
            c.total_spending     = float(request.form['total_spending'])
            c.last_purchase_date = datetime.strptime(
                request.form['last_purchase_date'], '%Y-%m-%d').date()
            c.status = request.form.get('status', c.status)
            c.notes  = request.form.get('notes', '').strip()

            X_raw = build_features(c.age, c.gender, c.purchase_freq,
                                   c.total_spending, c.last_purchase_date)
            res = predict_all(X_raw)
            c.churn_lr=res['churn_lr'];  c.churn_prob_lr=res['churn_prob_lr']
            c.churn_rf=res['churn_rf'];  c.churn_prob_rf=res['churn_prob_rf']
            c.churn_xgb=res['churn_xgb']; c.churn_prob_xgb=res['churn_prob_xgb']
            c.value_lr=res['value_lr'];  c.value_rf=res['value_rf']
            c.value_xgb=res['value_xgb']; c.segment=res['segment']
            c.shap_json=json.dumps(res['shap'])

            db.session.commit()
            log_action(f"Edited customer ID {cid}")
            flash('Customer updated successfully.', 'success')
            return redirect(url_for('customer_detail', cid=cid))
        except Exception as e:
            flash(f'Update error: {e}', 'danger')

    return render_template('customer_edit.html', c=c)


# ── Delete ─────────────────────────────────────────────────────────────────────
@app.route('/customer/<int:cid>/delete', methods=['POST'])
@login_required
def customer_delete(cid):
    c = my_customers().filter_by(id=cid).first_or_404()
    name = c.name
    db.session.delete(c)
    db.session.commit()
    log_action(f"Deleted customer: {name} (ID {cid})")
    flash(f'Customer "{name}" deleted.', 'success')
    return redirect(url_for('customers'))


# ════════════════════════════════════════════════════════════════════════════════
#  EXPORT / IMPORT  (per user)
# ════════════════════════════════════════════════════════════════════════════════
@app.route('/export/csv')
@login_required
def export_csv():
    customers = my_customers().order_by(Customer.created_at.desc()).all()
    si = io.StringIO()
    w  = csv.writer(si)
    w.writerow(['ID','Name','Age','Gender','Purchase Freq','Total Spending',
                'Last Purchase','Status','Churn LR','Churn RF','Churn XGB',
                'Value RF','Segment','Created At'])
    for c in customers:
        w.writerow([c.id, c.name, c.age, c.gender, c.purchase_freq,
                    c.total_spending, c.last_purchase_date, c.status,
                    c.churn_lr, c.churn_rf, c.churn_xgb,
                    c.value_rf, c.segment,
                    c.created_at.strftime('%Y-%m-%d %H:%M')])
    log_action('Exported CSV')
    return Response(si.getvalue(), mimetype='text/csv',
                    headers={'Content-Disposition':
                             f'attachment; filename={current_user.username}_customers.csv'})


@app.route('/import/csv', methods=['GET', 'POST'])
@login_required
def import_csv():
    results = []
    if request.method == 'POST':
        f = request.files.get('csv_file')
        if not f or not f.filename.endswith('.csv'):
            flash('Please upload a valid .csv file.', 'danger')
            return redirect(url_for('import_csv'))

        stream = io.StringIO(f.stream.read().decode('utf-8'))
        reader = csv.DictReader(stream)
        added, errors = 0, 0

        for row in reader:
            try:
                lpdate   = datetime.strptime(row['last_purchase_date'].strip(), '%Y-%m-%d').date()
                age      = int(row['age'])
                pfreq    = int(row['purchase_freq'])
                spending = float(row['total_spending'])
                gender   = row['gender'].strip()
                name     = row['name'].strip()

                X_raw  = build_features(age, gender, pfreq, spending, lpdate)
                res    = predict_all(X_raw)
                status = 'At Risk' if res['churn_rf'] == 'Yes' else 'Active'

                c = Customer(
                    user_id=current_user.id,     # ← tied to this user
                    name=name, age=age, gender=gender,
                    purchase_freq=pfreq, total_spending=spending,
                    last_purchase_date=lpdate, status=status,
                    churn_lr=res['churn_lr'],   churn_prob_lr=res['churn_prob_lr'],
                    churn_rf=res['churn_rf'],   churn_prob_rf=res['churn_prob_rf'],
                    churn_xgb=res['churn_xgb'], churn_prob_xgb=res['churn_prob_xgb'],
                    value_lr=res['value_lr'],   value_rf=res['value_rf'],
                    value_xgb=res['value_xgb'], segment=res['segment'],
                    shap_json=json.dumps(res['shap']),
                )
                db.session.add(c)
                added += 1
                results.append({'name': name, 'status': 'OK',
                                 'churn': res['churn_rf'], 'value': res['value_rf']})
            except Exception as e:
                errors += 1
                results.append({'name': row.get('name', '?'), 'status': f'Error: {e}'})

        db.session.commit()
        log_action(f'Bulk import: {added} added, {errors} errors')
        flash(f'Import complete: {added} added, {errors} skipped.',
              'success' if not errors else 'warning')

    return render_template('import_csv.html', results=results)


# ════════════════════════════════════════════════════════════════════════════════
#  PDF REPORT
# ════════════════════════════════════════════════════════════════════════════════
@app.route('/customer/<int:cid>/pdf')
@login_required
def export_pdf(cid):
    c         = my_customers().filter_by(id=cid).first_or_404()
    shap_data = json.loads(c.shap_json or '{}')

    buf    = io.BytesIO()
    doc    = SimpleDocTemplate(buf, pagesize=A4,
                               leftMargin=2*cm, rightMargin=2*cm,
                               topMargin=2*cm,  bottomMargin=2*cm)
    styles = getSampleStyleSheet()
    story  = []

    title_style = ParagraphStyle('T2', parent=styles['Title'], fontSize=20, spaceAfter=6)
    story.append(Paragraph("CustPredict — Customer Report", title_style))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%d %b %Y %H:%M')}  |  By: {current_user.username}",
                            styles['Normal']))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.grey))
    story.append(Spacer(1, 0.4*cm))

    story.append(Paragraph("Customer Profile", styles['Heading2']))
    profile_data = [
        ['Name', c.name], ['Age', str(c.age)], ['Gender', c.gender],
        ['Status', c.status],
        ['Purchase Freq', f"{c.purchase_freq}/yr"],
        ['Total Spending', f"${c.total_spending:.2f}"],
        ['Last Purchase', str(c.last_purchase_date)],
        ['Segment', c.segment],
    ]
    t = Table(profile_data, colWidths=[5*cm, 10*cm])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (0,-1), colors.HexColor('#0d1520')),
        ('TEXTCOLOR',  (0,0), (0,-1), colors.white),
        ('FONTNAME',   (0,0), (-1,-1), 'Helvetica'),
        ('FONTSIZE',   (0,0), (-1,-1), 10),
        ('ROWBACKGROUNDS', (1,0), (1,-1), [colors.whitesmoke, colors.white]),
        ('GRID', (0,0), (-1,-1), 0.4, colors.lightgrey),
        ('PADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(t)
    story.append(Spacer(1, 0.5*cm))

    story.append(Paragraph("Prediction Results", styles['Heading2']))
    pred_data = [
        ['Model', 'Churn', 'Churn Prob %', 'Customer Value'],
        ['Logistic Regression', c.churn_lr, f"{c.churn_prob_lr:.1f}%", c.value_lr],
        ['Random Forest',       c.churn_rf, f"{c.churn_prob_rf:.1f}%", c.value_rf],
        ['XGBoost',             c.churn_xgb, f"{c.churn_prob_xgb:.1f}%", c.value_xgb],
    ]
    t2 = Table(pred_data, colWidths=[5*cm, 3*cm, 3.5*cm, 3.5*cm])
    t2.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#4f8ef7')),
        ('TEXTCOLOR',  (0,0), (-1,0), colors.white),
        ('FONTNAME',   (0,0), (-1,-1), 'Helvetica'),
        ('FONTSIZE',   (0,0), (-1,-1), 10),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.whitesmoke, colors.white]),
        ('GRID', (0,0), (-1,-1), 0.4, colors.lightgrey),
        ('PADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(t2)
    story.append(Spacer(1, 0.5*cm))

    if shap_data:
        story.append(Paragraph("SHAP Feature Contributions", styles['Heading2']))
        shap_rows = [['Feature', 'SHAP Value', 'Direction']] + [
            [k, f"{v:+.4f}", '↑ Increases churn' if v > 0 else '↓ Reduces churn']
            for k, v in sorted(shap_data.items(), key=lambda x: abs(x[1]), reverse=True)
        ]
        t3 = Table(shap_rows, colWidths=[5*cm, 4*cm, 6*cm])
        t3.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#0d1520')),
            ('TEXTCOLOR',  (0,0), (-1,0), colors.white),
            ('FONTNAME',   (0,0), (-1,-1), 'Helvetica'),
            ('FONTSIZE',   (0,0), (-1,-1), 9),
            ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.whitesmoke, colors.white]),
            ('GRID', (0,0), (-1,-1), 0.4, colors.lightgrey),
            ('PADDING', (0,0), (-1,-1), 5),
        ]))
        story.append(t3)

    if c.notes:
        story.append(Spacer(1, 0.5*cm))
        story.append(Paragraph("Notes", styles['Heading2']))
        story.append(Paragraph(c.notes, styles['Normal']))

    doc.build(story)
    buf.seek(0)
    log_action(f'PDF report for customer ID {cid}')
    return Response(buf, mimetype='application/pdf',
                    headers={'Content-Disposition':
                             f'attachment; filename=customer_{cid}_report.pdf'})


# ════════════════════════════════════════════════════════════════════════════════
#  MODEL PERFORMANCE  (shared — same models for everyone)
# ════════════════════════════════════════════════════════════════════════════════
@app.route('/model-performance')
@login_required
def model_performance():
    metrics  = META['churn_metrics']
    feat_imp = META['feature_importance']
    return render_template('model_performance.html',
                           metrics=metrics,
                           feat_imp=feat_imp,
                           feat_names=list(feat_imp.keys()),
                           feat_vals=list(feat_imp.values()))


# ════════════════════════════════════════════════════════════════════════════════
#  ACCOUNT — profile, change password, delete account
# ════════════════════════════════════════════════════════════════════════════════
@app.route('/account', methods=['GET', 'POST'])
@login_required
def account():
    if request.method == 'POST':
        action = request.form.get('action')

        if action == 'update_profile':
            new_name  = request.form.get('full_name', '').strip()
            new_email = request.form.get('email', '').strip().lower()
            if not re.match(r'^[^@]+@[^@]+\.[^@]+$', new_email):
                flash('Invalid email address.', 'danger')
            elif new_email != current_user.email and User.query.filter_by(email=new_email).first():
                flash('That email is already in use.', 'danger')
            else:
                current_user.full_name = new_name
                current_user.email     = new_email
                db.session.commit()
                log_action('Updated profile')
                flash('Profile updated.', 'success')

        elif action == 'change_password':
            old = request.form.get('old_password', '')
            new = request.form.get('new_password', '')
            if not current_user.check_password(old):
                flash('Current password is incorrect.', 'danger')
            elif len(new) < 6:
                flash('New password must be at least 6 characters.', 'danger')
            else:
                current_user.set_password(new)
                db.session.commit()
                log_action('Changed password')
                flash('Password changed successfully.', 'success')

        elif action == 'delete_account':
            confirm = request.form.get('confirm_delete', '')
            if confirm != current_user.username:
                flash('Type your username exactly to confirm deletion.', 'danger')
            else:
                username = current_user.username
                logout_user()
                u = User.query.filter_by(username=username).first()
                db.session.delete(u)   # cascade deletes all their customers too
                db.session.commit()
                flash('Your account and all data have been permanently deleted.', 'info')
                return redirect(url_for('register'))

    total_customers = my_customers().count()
    return render_template('account.html', total_customers=total_customers)


# ════════════════════════════════════════════════════════════════════════════════
#  ADMIN ONLY — user list + audit log
# ════════════════════════════════════════════════════════════════════════════════
@app.route('/admin/users')
@login_required
@admin_required
def admin_users():
    all_users = User.query.order_by(User.created_at).all()
    user_stats = []
    for u in all_users:
        count = Customer.query.filter_by(user_id=u.id).count()
        user_stats.append({'user': u, 'customer_count': count})
    return render_template('admin_users.html', user_stats=user_stats)


@app.route('/admin/users/<int:uid>/delete', methods=['POST'])
@login_required
@admin_required
def admin_delete_user(uid):
    u = db.get_or_404(User, uid)
    if u.id == current_user.id:
        flash("You can't delete your own account here.", 'danger')
    else:
        name = u.username
        db.session.delete(u)
        db.session.commit()
        log_action(f'Admin deleted user: {name}')
        flash(f'User "{name}" and all their data deleted.', 'success')
    return redirect(url_for('admin_users'))


@app.route('/admin/audit-log')
@login_required
@admin_required
def audit_log():
    logs = AuditLog.query.order_by(AuditLog.timestamp.desc()).limit(300).all()
    return render_template('audit_log.html', logs=logs)


# ════════════════════════════════════════════════════════════════════════════════
#  REST API  (per-user data)
# ════════════════════════════════════════════════════════════════════════════════
@app.route('/api/customers')
@login_required
def api_customers():
    return jsonify([{
        'id': c.id, 'name': c.name, 'age': c.age, 'gender': c.gender,
        'purchase_freq': c.purchase_freq, 'total_spending': c.total_spending,
        'churn_rf': c.churn_rf, 'churn_prob_rf': c.churn_prob_rf,
        'value_rf': c.value_rf, 'segment': c.segment, 'status': c.status,
    } for c in my_customers().order_by(Customer.created_at.desc()).all()])


@app.route('/api/predict', methods=['POST'])
@login_required
@csrf.exempt
def api_predict():
    data = request.get_json()
    try:
        lpdate = datetime.strptime(data['last_purchase_date'], '%Y-%m-%d').date()
        X_raw  = build_features(int(data['age']), data['gender'],
                                 int(data['purchase_freq']),
                                 float(data['total_spending']), lpdate)
        return jsonify({'status': 'ok', 'predictions': predict_all(X_raw)})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400


@app.route('/api/stats')
@login_required
def api_stats():
    q     = my_customers()
    total = q.count()
    churn = q.filter_by(churn_rf='Yes').count()
    high  = q.filter_by(value_rf='High').count()
    avg   = db.session.query(db.func.avg(Customer.total_spending))\
              .filter_by(user_id=current_user.id).scalar() or 0
    return jsonify({'total': total, 'churn': churn,
                    'high_value': high, 'avg_spend': round(float(avg), 2)})


# ── DB init ────────────────────────────────────────────────────────────────────
def init_db():
    with app.app_context():
        db.create_all()
        print("[✓] Database tables created.")
        print("    No default users — everyone registers their own account.")
        print("    First person to register becomes admin automatically.")


if __name__ == '__main__':
    init_db()
    app.run(debug=True, port=5000)
