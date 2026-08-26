import { DesignWizard } from "@/components/design/DesignWizard";

export default function Home() {
  return (
    <main className="min-h-screen">
      <header className="border-b border-[var(--border)] px-6 py-4">
        <div className="flex items-center gap-3">
          <h1 className="text-xl font-semibold">Inference Design Planner</h1>
          <span className="rounded-full bg-[var(--accent)] px-2 py-0.5 text-xs text-[var(--primary)]">
            Preview
          </span>
        </div>
        <p className="mt-1 text-sm text-[var(--muted-foreground)]">
          Evidence-backed GPU inference deployment planning for OpenShift AI
        </p>
      </header>
      <DesignWizard />
    </main>
  );
}
