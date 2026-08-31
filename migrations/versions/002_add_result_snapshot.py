"""Add result_snapshot and completed_at columns

Revision ID: 002_add_result_snapshot
Revises: 001_initial_schema
Create Date: 2026-08-31 00:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "002_add_result_snapshot"
down_revision: str = "001_initial_schema"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    conn = op.get_bind()
    inspector = sa.inspect(conn)
    existing = [c["name"] for c in inspector.get_columns("design_sessions")]

    if "result_snapshot" not in existing:
        op.add_column(
            "design_sessions",
            sa.Column("result_snapshot", postgresql.JSONB(), nullable=True),
        )
    if "completed_at" not in existing:
        op.add_column(
            "design_sessions",
            sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        )

    op.create_index(
        "ix_design_sessions_completed_at",
        "design_sessions",
        ["completed_at"],
        if_not_exists=True,
    )


def downgrade() -> None:
    op.drop_index("ix_design_sessions_completed_at", table_name="design_sessions")
    op.drop_column("design_sessions", "completed_at")
    op.drop_column("design_sessions", "result_snapshot")
