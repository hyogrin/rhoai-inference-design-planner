"use client";

import {
  createContext,
  useContext,
  useState,
  useEffect,
  useCallback,
  useMemo,
  type ReactNode,
} from "react";

export type Language = "en" | "ko" | "zh" | "ja";

const LANG_KEY = "idp-language";
const SUPPORTED: Language[] = ["en", "ko", "zh", "ja"];

interface I18nContextValue {
  language: Language;
  setLanguage: (lang: Language) => void;
  t: (key: string, params?: Record<string, string | number>) => string;
}

const I18nContext = createContext<I18nContextValue>({
  language: "en",
  setLanguage: () => {},
  t: (key) => key,
});

export function I18nProvider({ children }: { children: ReactNode }) {
  const [language, setLanguageState] = useState<Language>("en");
  const [messages, setMessages] = useState<Record<string, string>>({});

  useEffect(() => {
    try {
      const stored = localStorage.getItem(LANG_KEY);
      if (stored && SUPPORTED.includes(stored as Language)) {
        setLanguageState(stored as Language);
      }
    } catch {
      // SSR or localStorage unavailable
    }
  }, []);

  useEffect(() => {
    fetch(`/locales/${language}.json`)
      .then((res) => res.json())
      .then((data) => setMessages(data))
      .catch(() => setMessages({}));
  }, [language]);

  const setLanguage = useCallback((lang: Language) => {
    setLanguageState(lang);
    try {
      localStorage.setItem(LANG_KEY, lang);
    } catch {
      // ignore
    }
  }, []);

  const t = useCallback(
    (key: string, params?: Record<string, string | number>) => {
      let value = messages[key] || key;
      if (params) {
        for (const [k, v] of Object.entries(params)) {
          value = value.replace(`{${k}}`, String(v));
        }
      }
      return value;
    },
    [messages],
  );

  const ctx = useMemo(
    () => ({ language, setLanguage, t }),
    [language, setLanguage, t],
  );

  return <I18nContext.Provider value={ctx}>{children}</I18nContext.Provider>;
}

export function useI18n() {
  return useContext(I18nContext);
}
