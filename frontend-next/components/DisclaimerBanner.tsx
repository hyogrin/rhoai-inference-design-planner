"use client";

import { useI18n } from "@/lib/i18n";

export default function DisclaimerBanner() {
  const { t } = useI18n();
  return (
    <div className="border-b border-amber-400 bg-amber-50 px-4 py-1.5 text-center text-xs text-amber-700">
      {t("topBanner.disclaimer")}
    </div>
  );
}
