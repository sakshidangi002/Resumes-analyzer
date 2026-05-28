/**
 * Web Push subscription helper.
 *
 * Flow on a logged-in user's first visit (per browser):
 *   1) Browser asks for `Notification.permission` (we wait until the user has
 *      been around for a few seconds — pestering at the login screen is rude).
 *   2) If granted: register `/sw.js`, fetch the server's VAPID public key,
 *      `pushManager.subscribe()`, and POST the subscription to /api/push/subscribe.
 *   3) Subsequent visits short-circuit if a subscription already exists.
 *
 * Gracefully no-ops when:
 *   - The browser doesn't support service workers / push (Safari < 16.4 etc.)
 *   - The page is served from an insecure origin (HTTP on a non-localhost host
 *     — browsers disable push there, by design).
 *   - The server hasn't configured a VAPID public key yet.
 *   - The user denies permission.
 */
import { push } from "../api/client";

function isSecureContext(): boolean {
  if (typeof window === "undefined") return false;
  // `window.isSecureContext` is true for https://* and http://localhost / 127.0.0.1.
  return !!(window as Window & { isSecureContext?: boolean }).isSecureContext;
}

function urlBase64ToUint8Array(base64String: string): Uint8Array {
  const padding = "=".repeat((4 - (base64String.length % 4)) % 4);
  const base64 = (base64String + padding).replace(/-/g, "+").replace(/_/g, "/");
  const rawData = atob(base64);
  const out = new Uint8Array(rawData.length);
  for (let i = 0; i < rawData.length; i += 1) {
    out[i] = rawData.charCodeAt(i);
  }
  return out;
}

let _ensurePromise: Promise<void> | null = null;

export function ensurePushSubscription(): Promise<void> {
  if (_ensurePromise) return _ensurePromise;
  _ensurePromise = (async () => {
    if (typeof window === "undefined") return;
    if (!("serviceWorker" in navigator)) return;
    if (!("PushManager" in window)) return;
    if (!("Notification" in window)) return;
    if (!isSecureContext()) {
      // Push API is disabled on insecure origins.
      // The frontend in-app banner still works, just no OS-level popup.
      // eslint-disable-next-line no-console
      console.info(
        "[push] Skipping Web Push setup: insecure origin (need HTTPS or localhost)."
      );
      return;
    }

    let permission: NotificationPermission = Notification.permission;
    if (permission === "default") {
      try {
        permission = await Notification.requestPermission();
      } catch {
        return;
      }
    }
    if (permission !== "granted") return;

    let keyRes;
    try {
      keyRes = await push.vapidPublicKey();
    } catch {
      return;
    }
    const vapidKey = keyRes?.data?.key || "";
    if (!vapidKey) {
      // eslint-disable-next-line no-console
      console.info("[push] Skipping Web Push setup: server has no VAPID key.");
      return;
    }

    let registration: ServiceWorkerRegistration;
    try {
      registration = await navigator.serviceWorker.register("/sw.js", {
        scope: "/",
      });
    } catch (e) {
      // eslint-disable-next-line no-console
      console.warn("[push] Service worker registration failed", e);
      return;
    }

    // Make sure the SW is active before we ask it to subscribe.
    if (registration.installing) {
      await new Promise<void>((resolve) => {
        registration.installing?.addEventListener("statechange", () => {
          if (registration.active) resolve();
        });
        setTimeout(resolve, 4000);
      });
    }

    let subscription: PushSubscription | null;
    try {
      subscription = await registration.pushManager.getSubscription();
    } catch {
      subscription = null;
    }

    if (!subscription) {
      try {
        // Cast: the DOM types want `BufferSource` backed by ArrayBuffer; our
        // helper returns Uint8Array<ArrayBufferLike> which is the same bytes
        // at runtime but TS narrowing rejects it.
        const appServerKey = urlBase64ToUint8Array(vapidKey)
          .buffer as ArrayBuffer;
        subscription = await registration.pushManager.subscribe({
          userVisibleOnly: true,
          applicationServerKey: appServerKey,
        });
      } catch (e) {
        // eslint-disable-next-line no-console
        console.warn("[push] pushManager.subscribe failed", e);
        return;
      }
    }

    if (!subscription) return;
    const json = subscription.toJSON();
    if (!json?.endpoint || !json?.keys?.p256dh || !json?.keys?.auth) return;

    try {
      await push.subscribe({
        endpoint: json.endpoint,
        keys: { p256dh: json.keys.p256dh, auth: json.keys.auth },
      });
    } catch (e) {
      // eslint-disable-next-line no-console
      console.warn("[push] Failed to register subscription with backend", e);
    }
  })();

  return _ensurePromise;
}

/**
 * Best-effort cleanup on logout. Unsubscribes the browser and tells the
 * server to forget this endpoint.
 */
export async function teardownPushSubscription(): Promise<void> {
  _ensurePromise = null;
  if (typeof window === "undefined") return;
  if (!("serviceWorker" in navigator)) return;
  try {
    const reg = await navigator.serviceWorker.getRegistration("/sw.js");
    if (!reg) return;
    const sub = await reg.pushManager.getSubscription();
    if (sub) {
      try {
        await push.unsubscribe(sub.endpoint);
      } catch {
        // ignore
      }
      try {
        await sub.unsubscribe();
      } catch {
        // ignore
      }
    }
  } catch {
    // ignore
  }
}
