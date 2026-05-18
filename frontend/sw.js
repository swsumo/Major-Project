const CACHE_NAME = 'fitai-v1';

const SHELL_FILES = [
  '/login.html',
  '/signup.html',
  '/dashboard.html',
  '/meals.html',
  '/exercise.html',
  '/progress.html',
  '/chat.html',
  '/profile.html',
  '/fl_dashboard.html',
  '/reset-password.html',
  '/frontend.html',
  '/static/auth.js',
  '/manifest.json'
];

// Install: cache the app shell
self.addEventListener('install', event => {
  event.waitUntil(
    caches.open(CACHE_NAME).then(cache => cache.addAll(SHELL_FILES))
  );
  self.skipWaiting();
});

// Activate: clear old caches
self.addEventListener('activate', event => {
  event.waitUntil(
    caches.keys().then(keys =>
      Promise.all(keys.filter(k => k !== CACHE_NAME).map(k => caches.delete(k)))
    )
  );
  self.clients.claim();
});

// Fetch: serve from cache, fall back to network
// API calls always go to network (never cached)
self.addEventListener('fetch', event => {
  const url = new URL(event.request.url);

  // Never cache API calls
  if (url.pathname.startsWith('/api/')) {
    event.respondWith(fetch(event.request));
    return;
  }

  event.respondWith(
    caches.match(event.request).then(cached => cached || fetch(event.request))
  );
});
