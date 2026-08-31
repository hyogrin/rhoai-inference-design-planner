"use client";

import Link from "next/link";
import { DesignWizard } from "@/components/design/DesignWizard";
import { useI18n, type Language } from "@/lib/i18n";
import { Globe } from "lucide-react";

const LANGUAGE_OPTIONS: { value: Language; label: string }[] = [
  { value: "en", label: "English" },
  { value: "ko", label: "한국어" },
  { value: "zh", label: "中文" },
  { value: "ja", label: "日本語" },
];

export default function Home() {
  const { language, setLanguage, t } = useI18n();

  return (
    <main className="min-h-screen">
      <header className="border-b border-[var(--border)] px-6 py-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <h1 className="text-xl font-semibold">{t("header.title")}</h1>
            <span className="rounded-full bg-[var(--accent)] px-2 py-0.5 text-xs text-[var(--primary)]">
              Preview
            </span>
          </div>
          <nav className="flex items-center gap-4">
            <Link
              href="/recommendations"
              className="text-sm font-medium text-[var(--muted-foreground)] hover:text-[var(--foreground)] transition-colors"
            >
              {t("header.savedRecommendations")}
            </Link>
            <div className="flex items-center gap-1.5 rounded-lg border border-[var(--border)] px-2 py-1.5">
              <Globe className="h-3.5 w-3.5 text-[var(--muted-foreground)]" />
              <select
                value={language}
                onChange={(e) => setLanguage(e.target.value as Language)}
                className="bg-transparent text-xs font-medium text-[var(--foreground)] outline-none cursor-pointer"
              >
                {LANGUAGE_OPTIONS.map((opt) => (
                  <option key={opt.value} value={opt.value}>
                    {opt.label}
                  </option>
                ))}
              </select>
            </div>
          </nav>
        </div>
        <p className="mt-1 text-sm text-[var(--muted-foreground)]">
          {t("header.subtitle")}
        </p>
      </header>
      <DesignWizard />
    </main>
  );
}
