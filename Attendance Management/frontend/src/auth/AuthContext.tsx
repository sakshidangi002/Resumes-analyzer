import React, { createContext, useContext, useState, useEffect, useCallback } from "react";
import { auth as authApi } from "../api/client";
import {
  startNotificationSocket,
  stopNotificationSocket,
} from "../lib/notificationsSocket";

export interface UserInfo {
  id: number;
  username: string;
  roles: string[];
  employee_id: number | null;
  employee_code?: string | null;
  designation?: string | null;
  official_email?: string;
  must_change_password?: boolean;
}

// NOTE: there is deliberately no `token` here. The session is an HttpOnly
// cookie the browser attaches automatically (axios sends it via
// withCredentials), so the app never holds the JWT. A leftover `token` field
// would always be null and any `if (!token)` check written against it would
// lock every user out — which is exactly what happened to PrivateRoute.
// "Is someone signed in?" is answered by `user`.
interface AuthContextType {
  user: UserInfo | null;
  loading: boolean;
  login: (username: string, password: string) => Promise<boolean>;
  signup: (username: string, password: string, official_email?: string) => Promise<void>;
  logout: () => void;
  hasRole: (...roles: string[]) => boolean;
  /** Clear the forced-password-change flag after /auth/change-password succeeds.
   *  Without this the flag stays true in context and PrivateRoute bounces the
   *  user straight back to /change-password, stranding them on that screen.
   *  The refreshed session cookie (with pwd_change cleared, which is what the
   *  Resume Analyzer gate reads) is set by the endpoint itself. */
  markPasswordChanged: () => void;
}

const AuthContext = createContext<AuthContextType | null>(null);

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [user, setUser] = useState<UserInfo | null>(null);
  const [loading, setLoading] = useState(true);

  const loadUser = useCallback(async () => {
    // Always probe. The session cookie is HttpOnly, so there is nothing the
    // client can inspect first — asking the server is the only way to know.
    // A 401 here means "nobody is logged in", which is a normal answer and is
    // why /auth/me is excluded from the redirect-on-401 interceptor.
    try {
      // Race against a 10s timeout so we never hang on startup
      const timeoutPromise = new Promise<never>((_, reject) =>
        setTimeout(() => reject(new Error("timeout")), 10000)
      );
      const { data } = await Promise.race([authApi.me(), timeoutPromise]);
      setUser({
        id: data.id,
        username: data.username,
        roles: data.roles || [],
        employee_id: data.employee_id ?? null,
        employee_code: data.employee_code ?? null,
        designation: data.designation ?? null,
        official_email: data.official_email,
        must_change_password: data.must_change_password,
      });
      startNotificationSocket();
    } catch {
      setUser(null);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadUser();
  }, [loadUser]);

  const login = async (username: string, password: string) => {
    const { data } = await authApi.login(username, password);
    setUser({
      id: data.user_id,
      username: data.username,
      roles: data.roles || [],
      employee_id: data.employee_id ?? null,
      employee_code: data.employee_code ?? null,
      designation: data.designation ?? null,
      must_change_password: data.must_change_password,
    });
    startNotificationSocket();
    return Boolean(data.must_change_password);
  };

  const signup = async (username: string, password: string, official_email?: string) => {
    const { data } = await authApi.signup(username, password, official_email);
    setUser({
      id: data.user_id,
      username: data.username,
      roles: data.roles || [],
      employee_id: data.employee_id ?? null,
      employee_code: data.employee_code ?? null,
      designation: data.designation ?? null,
    });
    startNotificationSocket();
  };

  const logout = () => {
    // Fire-and-forget cleanup of any Web Push subscription registered for this
    // user. Don't await — logout must be instant from the user's POV.
    import("../lib/push")
      .then((m) => m.teardownPushSubscription())
      .catch(() => {});
    stopNotificationSocket();
    void authApi.logout().catch(() => undefined);
    setUser(null);
  };

  const markPasswordChanged = () => {
    setUser((prev) => (prev ? { ...prev, must_change_password: false } : prev));
  };

  const hasRole = (...roles: string[]) => {
    if (!user?.roles?.length) return false;
    return roles.some((r) => user.roles.includes(r));
  };

  return (
    <AuthContext.Provider
      value={{ user, loading, login, signup, logout, hasRole, markPasswordChanged }}
    >
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth must be used within AuthProvider");
  return ctx;
}
