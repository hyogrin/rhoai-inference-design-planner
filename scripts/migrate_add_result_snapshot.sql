-- Migration: Add result_snapshot and completed_at columns to design_sessions
-- Run: psql $DATABASE_URL -f scripts/migrate_add_result_snapshot.sql

ALTER TABLE design_sessions
  ADD COLUMN IF NOT EXISTS result_snapshot JSONB,
  ADD COLUMN IF NOT EXISTS completed_at TIMESTAMPTZ;

CREATE INDEX IF NOT EXISTS ix_design_sessions_completed_at
  ON design_sessions (completed_at DESC NULLS LAST)
  WHERE completed_at IS NOT NULL;
