"""Initial schema setup: settings, basic model_meta, consent, audit

Revision ID: 0001_initial
Revises:
Create Date: 2026-11-12 10:00:00.000000

"""

import sqlalchemy as sa

from alembic import op

revision = "0001_initial"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "settings",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("key", sa.String(length=255), nullable=False, unique=True),
        sa.Column("value", sa.Text(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=True),
    )

    op.create_table(
        "model_meta",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("model_id", sa.String(length=255), nullable=False, unique=True),
        sa.Column("engine_id", sa.String(length=64), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("sha256", sa.String(length=64), nullable=False),
        sa.Column("size_bytes", sa.Integer(), nullable=False),
        sa.Column("license_type", sa.String(length=64), nullable=False),
        sa.Column("consent_required", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=True),
    )

    op.create_table(
        "consent",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("model_id", sa.String(length=255), nullable=False),
        sa.Column("attested_by", sa.String(length=255), nullable=False),
        sa.Column("attested_at", sa.DateTime(), nullable=False),
        sa.Column("license_type", sa.String(length=64), nullable=False),
        sa.Column("notes", sa.Text(), nullable=True),
    )

    op.create_table(
        "audit",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("event_type", sa.String(length=64), nullable=False),
        sa.Column("payload", sa.Text(), nullable=False),
        sa.Column("occurred_at", sa.DateTime(), nullable=True),
    )


def downgrade() -> None:
    op.drop_table("audit")
    op.drop_table("consent")
    op.drop_table("model_meta")
    op.drop_table("settings")
