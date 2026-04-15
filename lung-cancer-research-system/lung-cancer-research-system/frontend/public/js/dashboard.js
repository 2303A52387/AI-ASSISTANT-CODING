// ─── LungAI Research System — Dashboard JS ───────────────

document.addEventListener('DOMContentLoaded', () => {
  initAnimations();
  initTooltips();
});

// Fade-in KPI cards on load
function initAnimations() {
  const cards = document.querySelectorAll('.kpi-card, .section-card');
  cards.forEach((card, i) => {
    card.style.opacity = '0';
    card.style.transform = 'translateY(16px)';
    card.style.transition = 'opacity 0.4s ease, transform 0.4s ease';
    setTimeout(() => {
      card.style.opacity = '1';
      card.style.transform = 'translateY(0)';
    }, 60 + i * 40);
  });
}

function initTooltips() {
  const tooltipEls = document.querySelectorAll('[data-tooltip]');
  tooltipEls.forEach(el => {
    el.title = el.dataset.tooltip;
  });
}

// Global fetch helper with loading state
async function apiPost(url, data, btnEl) {
  if (btnEl) { btnEl.disabled = true; btnEl.innerHTML += ' <span class="spinner-sm"></span>'; }
  try {
    const res = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data)
    });
    return await res.json();
  } finally {
    if (btnEl) { btnEl.disabled = false; btnEl.querySelector('.spinner-sm')?.remove(); }
  }
}

// Number counter animation
function animateCounter(el, target, duration = 1000, suffix = '') {
  const start = 0;
  const startTime = performance.now();
  function update(currentTime) {
    const elapsed = currentTime - startTime;
    const progress = Math.min(elapsed / duration, 1);
    const eased = 1 - Math.pow(1 - progress, 3);
    el.textContent = (start + (target - start) * eased).toFixed(2) + suffix;
    if (progress < 1) requestAnimationFrame(update);
  }
  requestAnimationFrame(update);
}

// Notification toast
function showToast(message, type = 'success') {
  const toast = document.createElement('div');
  toast.className = `toast-msg toast-${type}`;
  toast.innerHTML = `<i class="fas fa-${type === 'success' ? 'check-circle' : 'exclamation-circle'}"></i> ${message}`;
  toast.style.cssText = `
    position: fixed; bottom: 1.5rem; right: 1.5rem;
    background: ${type === 'success' ? 'rgba(34,197,94,0.15)' : 'rgba(239,68,68,0.15)'};
    border: 1px solid ${type === 'success' ? 'rgba(34,197,94,0.4)' : 'rgba(239,68,68,0.4)'};
    color: ${type === 'success' ? '#22c55e' : '#ef4444'};
    padding: 0.75rem 1.25rem; border-radius: 8px;
    font-size: 0.85rem; font-weight: 500;
    z-index: 9999; display: flex; gap: 8px; align-items: center;
    animation: slideIn 0.3s ease;
  `;
  document.body.appendChild(toast);
  setTimeout(() => toast.remove(), 3500);
}

window.showToast = showToast;
window.apiPost = apiPost;
window.animateCounter = animateCounter;
