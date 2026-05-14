// Shared theme toggle — included in every page
(function () {
  const saved = localStorage.getItem('retina-theme') || 'dark';
  document.documentElement.setAttribute('data-theme', saved);
  const btn = document.getElementById('themeToggle');
  if (btn) btn.textContent = saved === 'dark' ? '🌙' : '☀️';
})();

function toggleTheme() {
  const html = document.documentElement;
  const next = html.dataset.theme === 'dark' ? 'light' : 'dark';
  html.dataset.theme = next;
  localStorage.setItem('retina-theme', next);
  const btn = document.getElementById('themeToggle');
  if (btn) btn.textContent = next === 'dark' ? '🌙' : '☀️';
}

function showToast(msg) {
  let t = document.getElementById('toast');
  if (!t) {
    t = document.createElement('div');
    t.id = 'toast';
    t.className = 'toast';
    document.body.appendChild(t);
  }
  t.textContent = msg;
  t.classList.add('show');
  setTimeout(() => t.classList.remove('show'), 3200);
}

// ── Mobile hamburger menu ─────────────────
function toggleMenu() {
  const links = document.querySelector('.nav-links');
  if (links) links.classList.toggle('open');
}

// Close menu when clicking a link
document.addEventListener('click', function(e) {
  const links = document.querySelector('.nav-links');
  const hamburger = document.querySelector('.hamburger');
  if (links && links.classList.contains('open') &&
      !links.contains(e.target) && !hamburger.contains(e.target)) {
    links.classList.remove('open');
  }
});

// ── Scroll-triggered fade-in animations ───
document.addEventListener('DOMContentLoaded', function() {
  const faders = document.querySelectorAll('.fade-in');
  if (faders.length === 0) return;

  const observer = new IntersectionObserver(function(entries) {
    entries.forEach(function(entry) {
      if (entry.isIntersecting) {
        entry.target.classList.add('visible');
        observer.unobserve(entry.target);
      }
    });
  }, { threshold: 0.15 });

  faders.forEach(function(el) { observer.observe(el); });
});