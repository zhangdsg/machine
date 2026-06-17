/* ─── Theme ─── */
const html = document.documentElement;
const themeBtn = document.getElementById('themeBtn');

function setTheme(mode) {
  html.setAttribute('data-theme', mode);
  themeBtn.textContent = mode === 'dark' ? '🌙' : '☀️';
  localStorage.setItem('theme', mode);
}

function toggleTheme() {
  setTheme(html.getAttribute('data-theme') === 'dark' ? 'light' : 'dark');
}

const saved = localStorage.getItem('theme');
if (saved) setTheme(saved);

themeBtn.addEventListener('click', toggleTheme);

/* ─── Mobile Nav ─── */
const mobileBtn = document.getElementById('mobileBtn');
const navLinks = document.getElementById('navLinks');

mobileBtn.addEventListener('click', () => {
  navLinks.classList.toggle('open');
});

navLinks.querySelectorAll('a').forEach(a => {
  a.addEventListener('click', () => navLinks.classList.remove('open'));
});

/* ─── Active Nav Link ─── */
const sections = document.querySelectorAll('.section');
const navAnchors = navLinks.querySelectorAll('a[href^="#"]');

function updateActiveNav() {
  let current = '';
  sections.forEach(s => {
    const top = s.offsetTop - 120;
    if (window.scrollY >= top) current = s.id;
  });
  navAnchors.forEach(a => {
    a.classList.toggle('active', a.getAttribute('href') === '#' + current);
  });
}

window.addEventListener('scroll', updateActiveNav, { passive: true });

/* ─── Lightbox ─── */
const lightbox = document.getElementById('lightbox');
const lightboxImg = document.getElementById('lightboxImg');

function openLightbox(src) {
  lightboxImg.src = src;
  lightbox.classList.add('open');
  document.body.style.overflow = 'hidden';
}

function closeLightbox() {
  lightbox.classList.remove('open');
  document.body.style.overflow = '';
}

lightbox.addEventListener('click', closeLightbox);

document.addEventListener('keydown', e => {
  if (e.key === 'Escape') closeLightbox();
});

/* ─── Scroll Reveal ─── */
const revealObserver = new IntersectionObserver(entries => {
  entries.forEach(entry => {
    if (entry.isIntersecting) {
      entry.target.classList.add('visible');
    }
  });
}, { threshold: 0.08, rootMargin: '0px 0px -40px 0px' });

document.querySelectorAll('.reveal').forEach(el => revealObserver.observe(el));

/* ─── Loading Bar ─── */
const bar = document.getElementById('loadingBar');
let loaded = 0;

function advance() {
  loaded += Math.random() * 20 + 5;
  if (loaded > 90) loaded = 90;
  bar.style.width = loaded + '%';
}

const interval = setInterval(advance, 200);

window.addEventListener('load', () => {
  clearInterval(interval);
  bar.style.width = '100%';
  setTimeout(() => {
    bar.style.transition = 'opacity 0.6s';
    bar.style.opacity = '0';
  }, 300);
});

/* ─── Smooth Scroll for Anchor Links (fallback) ─── */
document.querySelectorAll('a[href^="#"]').forEach(a => {
  a.addEventListener('click', e => {
    const id = a.getAttribute('href');
    if (id === '#') return;
    const target = document.querySelector(id);
    if (target) {
      e.preventDefault();
      target.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  });
});
