/* Softwiz HRMS service worker — handles Web Push from the 5 PM IST DSR reminder
 * (and any future server-driven notification). Kept dependency-free.
 *
 * Payload shape pushed by the backend (see app/services/web_push_service.py):
 *   { title: string, body: string, url?: string, tag?: string }
 */

/* eslint-disable no-restricted-globals */

self.addEventListener('install', () => {
  // Activate this SW as soon as it's installed so a new build takes effect
  // without waiting for every tab to close.
  self.skipWaiting();
});

self.addEventListener('activate', (event) => {
  event.waitUntil(self.clients.claim());
});

self.addEventListener('push', (event) => {
  let payload = { title: 'Notification', body: '', url: '/', tag: 'softwiz' };
  try {
    if (event.data) {
      payload = Object.assign(payload, event.data.json());
    }
  } catch (_e) {
    try {
      payload.body = event.data ? event.data.text() : '';
    } catch (_e2) {}
  }

  const title = payload.title || 'Notification';
  const options = {
    body: payload.body || '',
    icon: '/favicon.ico',
    badge: '/favicon.ico',
    tag: payload.tag || 'softwiz',
    renotify: true,
    requireInteraction: true,
    data: { url: payload.url || '/' },
  };

  // Show the OS popup AND nudge any open Softwiz tab so the bell badge
  // and inbox refresh instantly — without waiting for the 60 s poll.
  event.waitUntil(
    Promise.all([
      self.registration.showNotification(title, options),
      self.clients
        .matchAll({ type: 'window', includeUncontrolled: true })
        .then((clients) => {
          for (const c of clients) {
            try {
              c.postMessage({ type: 'softwiz:notif-refresh', tag: options.tag });
            } catch (_e) {}
          }
        })
        .catch(() => {}),
    ]),
  );
});

self.addEventListener('notificationclick', (event) => {
  event.notification.close();
  const url = (event.notification.data && event.notification.data.url) || '/';

  event.waitUntil(
    self.clients
      .matchAll({ type: 'window', includeUncontrolled: true })
      .then((windowClients) => {
        for (const client of windowClients) {
          try {
            const clientUrl = new URL(client.url);
            // If we already have a tab open on our origin, focus + navigate it.
            if (clientUrl.origin === self.location.origin && 'focus' in client) {
              if ('navigate' in client) {
                client.navigate(url).catch(() => {});
              }
              return client.focus();
            }
          } catch (_e) {}
        }
        // Otherwise open a fresh tab.
        if (self.clients.openWindow) {
          return self.clients.openWindow(url);
        }
        return null;
      })
  );
});
