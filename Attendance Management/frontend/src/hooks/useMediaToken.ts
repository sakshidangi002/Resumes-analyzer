import { useCallback, useEffect, useRef, useState } from "react";
import { cameras as camerasApi } from "../api/client";

// The backend issues media tokens with a ~120s lifetime. Refresh well before
// that so a page left open keeps rendering: an <img> that starts a request with
// an expired token just shows a broken image.
//
// This only affects NEW connections. An MJPEG stream already running is not
// interrupted by its token expiring — the server checks at connect time only.
const REFRESH_INTERVAL_MS = 90_000;

/**
 * Keeps a short-lived camera media token available for <img src> URLs.
 *
 * Returns an empty string until the first token arrives; callers should hold
 * off rendering the <img> until then, otherwise the browser fires a request
 * with no token and paints a broken image.
 */
export function useMediaToken(enabled: boolean = true) {
  const [token, setToken] = useState("");
  const [error, setError] = useState(false);
  // Guards against a response from a previous `enabled` cycle landing after
  // the component has moved on and overwriting a newer token.
  const generationRef = useRef(0);

  const refresh = useCallback(async () => {
    const generation = generationRef.current;
    try {
      const { data } = await camerasApi.mediaToken();
      if (generation === generationRef.current) {
        setToken(data.token);
        setError(false);
      }
    } catch {
      if (generation === generationRef.current) setError(true);
    }
  }, []);

  useEffect(() => {
    generationRef.current += 1;
    if (!enabled) {
      setToken("");
      return;
    }
    void refresh();
    const timer = window.setInterval(() => void refresh(), REFRESH_INTERVAL_MS);
    return () => window.clearInterval(timer);
  }, [enabled, refresh]);

  return { mediaToken: token, mediaTokenError: error, refreshMediaToken: refresh };
}
