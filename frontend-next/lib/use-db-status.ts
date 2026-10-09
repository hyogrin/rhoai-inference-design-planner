"use client";

import { useEffect, useState } from "react";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:7001";

export function useDbStatus() {
  const [dbAvailable, setDbAvailable] = useState<boolean | null>(null);

  useEffect(() => {
    let cancelled = false;
    async function check() {
      try {
        const res = await fetch(`${API_BASE}/api/v1/health`);
        if (!res.ok) {
          if (!cancelled) setDbAvailable(false);
          return;
        }
        const data = await res.json();
        if (!cancelled) setDbAvailable(data.db_available === true);
      } catch {
        if (!cancelled) setDbAvailable(false);
      }
    }
    check();
    return () => {
      cancelled = true;
    };
  }, []);

  return dbAvailable;
}
