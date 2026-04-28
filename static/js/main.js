/* main.js – CustPredict v2 */

// ── Sidebar toggle (mobile) ────────────────────────────────────────────────
const toggleBtn = document.getElementById('sidebarToggle');
const sidebar   = document.getElementById('sidebar');

if (toggleBtn && sidebar) {
  toggleBtn.addEventListener('click', () => sidebar.classList.toggle('open'));
  document.addEventListener('click', (e) => {
    if (window.innerWidth < 769 && sidebar.classList.contains('open') &&
        !sidebar.contains(e.target) && e.target !== toggleBtn) {
      sidebar.classList.remove('open');
    }
  });
}

// ── Auto-dismiss alerts after 5s ──────────────────────────────────────────
document.querySelectorAll('.alert.alert-dismissible').forEach(el => {
  setTimeout(() => {
    try { bootstrap.Alert.getOrCreateInstance(el).close(); } catch(e) {}
  }, 5000);
});

// ── Set today as max for date fields ──────────────────────────────────────
document.querySelectorAll('[name="last_purchase_date"]').forEach(el => {
  el.setAttribute('max', new Date().toISOString().split('T')[0]);
});

// ── Animate KPI counters on scroll into view ───────────────────────────────
function animateCounter(el) {
  const raw    = el.textContent.trim();
  const prefix = raw.includes('$') ? '$' : '';
  const target = parseFloat(raw.replace(/[^0-9.]/g, ''));
  if (isNaN(target) || target === 0) return;
  const isDecimal = raw.includes('.');
  const duration  = 900;
  const step      = target / (duration / 16);
  let   current   = 0;
  const timer = setInterval(() => {
    current += step;
    if (current >= target) { current = target; clearInterval(timer); }
    el.textContent = prefix + (isDecimal ? current.toFixed(2) : Math.floor(current));
  }, 16);
}

const obs = new IntersectionObserver(entries => {
  entries.forEach(e => {
    if (e.isIntersecting) {
      animateCounter(e.target);
      obs.unobserve(e.target);
    }
  });
}, { threshold: 0.5 });

document.querySelectorAll('.kpi-value').forEach(el => obs.observe(el));

// ── Prob bars: animate width on load ──────────────────────────────────────
document.querySelectorAll('.prob-bar').forEach(bar => {
  const target = bar.style.width;
  bar.style.width = '0%';
  setTimeout(() => { bar.style.width = target; }, 200);
});

// ── SHAP bars: animate on load ─────────────────────────────────────────────
document.querySelectorAll('.shap-bar').forEach(bar => {
  const target = bar.style.width;
  bar.style.width = '0%';
  setTimeout(() => { bar.style.width = target; }, 300);
});

// ── File input label update ────────────────────────────────────────────────
document.querySelectorAll('input[type="file"]').forEach(input => {
  input.addEventListener('change', () => {
    if (input.files[0]) {
      const label = input.closest('.mb-3, .mb-4')?.querySelector('.form-label-custom');
      if (label) label.textContent = `✓ ${input.files[0].name}`;
    }
  });
});

// ── Confirm delete shortcut ────────────────────────────────────────────────
document.querySelectorAll('[data-confirm]').forEach(el => {
  el.addEventListener('click', e => {
    if (!confirm(el.dataset.confirm)) e.preventDefault();
  });
});
