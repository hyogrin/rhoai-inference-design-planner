import type { Metadata } from "next";
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
      <body className="min-h-screen antialiased">{children}</body>
    </html>
  );
}
