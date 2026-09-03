import type { Metadata } from "next";
import { I18nProvider } from "@/lib/i18n";
import "./globals.css";

export const metadata: Metadata = {
  title: "Inference Design Planner",
  description:
    "Evidence-backed inference design planning for Red Hat OpenShift AI",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="min-h-screen antialiased">
        <div className="border-b border-red-500 bg-red-50 px-4 py-1.5 text-center text-sm font-semibold text-red-600">
          ⚠️ INTERNAL USE ONLY — DO NOT SHARE EXTERNALLY
        </div>
        <I18nProvider>{children}</I18nProvider>
      </body>
    </html>
  );
}
