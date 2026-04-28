"""
init_db.py  –  v3
-----------------
Creates the SQLite database tables only.
NO default users are created — everyone registers their own account.
The first person to register at /register becomes admin automatically.
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from app import app, db

def init():
    with app.app_context():
        db.create_all()
        print("[✓] Database tables created  →  instance/customers.db")
        print()
        print("  No default users are seeded.")
        print("  → Open http://127.0.0.1:5000/register")
        print("  → The FIRST person to register becomes admin.")
        print("  → Every user sees only their own customer data.")
        print()
        print("  Run the app:  python app.py")

if __name__ == '__main__':
    init()
