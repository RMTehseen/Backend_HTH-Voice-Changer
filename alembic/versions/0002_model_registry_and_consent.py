"""Expand model_meta, consent, and audit tables for Phase 7 registry

Revision ID: 0002_model_registry_and_consent
Revises: 0001_initial
Create Date: 2026-11-12 12:00:00.000000

"""

import sqlalchemy as sa

from alembic import op

revision = "0002_model_registry_and_consent"
down_revision = "0001_initial"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Add columns to model_meta
    with op.batch_alter_table("model_meta") as batch_op:
        batch_op.add_column(
            sa.Column(
                "model_format", sa.String(length=64), server_default="unknown", nullable=False
            )
        )
        batch_op.add_column(
            sa.Column("version", sa.String(length=32), server_default="1.0.0", nullable=False)
        )
        batch_op.add_column(sa.Column("description", sa.Text(), server_default="", nullable=False))
        batch_op.add_column(
            sa.Column("manifest_json", sa.Text(), server_default="{}", nullable=False)
        )
        batch_op.add_column(
            sa.Column("asset_list_json", sa.Text(), server_default="[]", nullable=False)
        )
        batch_op.add_column(
            sa.Column("sample_rate", sa.Integer(), server_default="16000", nullable=False)
        )
        batch_op.add_column(
            sa.Column(
                "license_identifier",
                sa.String(length=64),
                server_default="Proprietary",
                nullable=False,
            )
        )
        batch_op.add_column(
            sa.Column(
                "license_source_url", sa.String(length=1024), server_default="", nullable=False
            )
        )
        batch_op.add_column(
            sa.Column(
                "source_type",
                sa.String(length=64),
                server_default="imported_archive",
                nullable=False,
            )
        )
        batch_op.add_column(
            sa.Column("owner_consent_evidence", sa.Text(), server_default="", nullable=False)
        )
        batch_op.add_column(
            sa.Column(
                "commercial_eligible", sa.Boolean(), server_default=sa.text("0"), nullable=False
            )
        )
        batch_op.add_column(
            sa.Column(
                "is_public_figure_detected",
                sa.Boolean(),
                server_default=sa.text("0"),
                nullable=False,
            )
        )
        batch_op.add_column(
            sa.Column(
                "compatibility_status",
                sa.String(length=64),
                server_default="consent_required",
                nullable=False,
            )
        )
        batch_op.add_column(sa.Column("install_path", sa.Text(), server_default="", nullable=False))
        batch_op.add_column(
            sa.Column("is_active", sa.Boolean(), server_default=sa.text("0"), nullable=False)
        )
        batch_op.add_column(
            sa.Column("updated_at", sa.DateTime(), server_default=sa.func.now(), nullable=False)
        )
        batch_op.create_index("ix_model_meta_model_id", ["model_id"])

    # Expand consent table
    with op.batch_alter_table("consent") as batch_op:
        batch_op.add_column(
            sa.Column("is_own_voice", sa.Boolean(), server_default=sa.text("0"), nullable=False)
        )
        batch_op.add_column(
            sa.Column(
                "has_written_consent", sa.Boolean(), server_default=sa.text("0"), nullable=False
            )
        )
        batch_op.add_column(
            sa.Column(
                "has_commercial_rights", sa.Boolean(), server_default=sa.text("0"), nullable=False
            )
        )
        batch_op.add_column(
            sa.Column(
                "no_celebrity_likeness", sa.Boolean(), server_default=sa.text("0"), nullable=False
            )
        )
        batch_op.add_column(
            sa.Column("license_accepted", sa.Boolean(), server_default=sa.text("0"), nullable=False)
        )
        batch_op.create_index("ix_consent_model_id", ["model_id"])

    # Add model_id to audit table
    with op.batch_alter_table("audit") as batch_op:
        batch_op.add_column(sa.Column("model_id", sa.String(length=255), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("audit") as batch_op:
        batch_op.drop_column("model_id")

    with op.batch_alter_table("consent") as batch_op:
        batch_op.drop_index("ix_consent_model_id")
        batch_op.drop_column("license_accepted")
        batch_op.drop_column("no_celebrity_likeness")
        batch_op.drop_column("has_commercial_rights")
        batch_op.drop_column("has_written_consent")
        batch_op.drop_column("is_own_voice")

    with op.batch_alter_table("model_meta") as batch_op:
        batch_op.drop_index("ix_model_meta_model_id")
        batch_op.drop_column("updated_at")
        batch_op.drop_column("is_active")
        batch_op.drop_column("install_path")
        batch_op.drop_column("compatibility_status")
        batch_op.drop_column("is_public_figure_detected")
        batch_op.drop_column("commercial_eligible")
        batch_op.drop_column("owner_consent_evidence")
        batch_op.drop_column("source_type")
        batch_op.drop_column("license_source_url")
        batch_op.drop_column("license_identifier")
        batch_op.drop_column("sample_rate")
        batch_op.drop_column("asset_list_json")
        batch_op.drop_column("manifest_json")
        batch_op.drop_column("description")
        batch_op.drop_column("version")
        batch_op.drop_column("model_format")
