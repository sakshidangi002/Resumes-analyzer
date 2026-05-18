import React, { createContext, useContext, useState, useEffect, useCallback } from "react";
import { auth as authApi } from "../api/client";

export interface UserInfo {
  id: number;
  username: string;
  roles: string[];
  employee_id: number | null;
  employee_code?: string | null;
  designation?: string | null;
  official_email?: string;
}

interface AuthContextType {
  user: UserInfo | null;
  token: string | null;
  loading: boolean;
  login: (username: string, password: string) => Promise<void>;
  signup: (username: string, password: string, official_email?: string) => Promise<void>;
  logout: () => void;
  hasRole: (...roles: string[]) => boolean;
}

const AuthContext = createContext<AuthContextType | null>(null);

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [user, setUser] = useState<UserInfo | null>(null);
  const [token, setToken] = useState<string | null>(localStorage.getItem("token"));
  const [loading, setLoading] = useState(true);

  const loadUser = useCallback(async () => {
    if (!token) {
      setUser(null);
      setLoading(false);
      return;
    }
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
      });
    } catch {
      setToken(null);
      setUser(null);
      localStorage.removeItem("token");
      localStorage.removeItem("user");
    } finally {
      setLoading(false);
    }
  }, [token]);

  useEffect(() => {
    loadUser();
  }, [loadUser]);

  const login = async (username: string, password: string) => {
    const { data } = await authApi.login(username, password);
    localStorage.setItem("token", data.access_token);
    setToken(data.access_token);
    setUser({
      id: data.user_id,
      username: data.username,
      roles: data.roles || [],
      employee_id: data.employee_id ?? null,
      employee_code: data.employee_code ?? null,
      designation: data.designation ?? null,
    });
  };

  const signup = async (username: string, password: string, official_email?: string) => {
    const { data } = await authApi.signup(username, password, official_email);
    localStorage.setItem("token", data.access_token);
    setToken(data.access_token);
    setUser({
      id: data.user_id,
      username: data.username,
      roles: data.roles || [],
      employee_id: data.employee_id ?? null,
      employee_code: data.employee_code ?? null,
      designation: data.designation ?? null,
    });
  };

  const logout = () => {
    localStorage.removeItem("token");
    localStorage.removeItem("user");
    setToken(null);
    setUser(null);
  };

  const hasRole = (...roles: string[]) => {
    if (!user?.roles?.length) return false;
    return roles.some((r) => user.roles.includes(r));
  };

  return (
    <AuthContext.Provider value={{ user, token, loading, login, signup, logout, hasRole }}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth must be used within AuthProvider");
  return ctx;
}
