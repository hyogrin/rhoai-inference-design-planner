"""Initial schema - design_sessions table

Revision ID: 001_initial_schema
Revises:
Create Date: 2025-01-01 00:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "001_initial_schema"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "design_sessions",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("title", sa.String(255), nullable=True),
        sa.Column("status", sa.String(50), nullable=False, server_default="intake"),
        sa.Column("model_repo_id", sa.String(255), nullable=True),
        sa.Column("model_revision", sa.String(255), nullable=True),
        sa.Column("current_step", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("state_snapshot", postgresql.JSONB(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column("version", sa.Integer(), nullable=False, server_default="1"),
    )

    op.create_index(
        "ix_design_sessions_status", "design_sessions", ["status"]
    )
    op.create_index(
        "ix_design_sessions_created_at", "design_sessions", ["created_at"]
    )
    op.create_index(
        "ix_design_sessions_model_repo_id", "design_sessions", ["model_repo_id"]
    )


def downgrade() -> None:
    op.drop_index("ix_design_sessions_model_repo_id", table_name="design_sessions")
    op.drop_index("ix_design_sessions_created_at", table_name="design_sessions")
    op.drop_index("ix_design_sessions_status", table_name="design_sessions")
    op.drop_table("design_sessions")
