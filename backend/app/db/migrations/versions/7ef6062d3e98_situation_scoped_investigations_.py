"""situation-scoped investigations, proposals and tickets

Revision ID: 7ef6062d3e98
Revises: 071427df1cc0
Create Date: 2026-09-21 01:35:57.292265

Hand-corrected after autogeneration. Alembic emitted the new NOT NULL columns
with no server default, which fails immediately on any table that already has
rows - and the deployed database has tickets in it. Each column is therefore
added with a server default, backfilled, and only then constrained.

`subject_id` is backfilled from `order_id`, which is correct: every row that
exists before this migration is an order investigation.
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = '7ef6062d3e98'
down_revision = '071427df1cc0'
branch_labels = None
depends_on = None

_TABLES = ("investigations", "proposals", "tickets")


def upgrade() -> None:
    for table in _TABLES:
        op.add_column(
            table,
            sa.Column(
                "subject_type",
                sa.String(length=16),
                nullable=False,
                server_default="order",
            ),
        )
        # Added nullable, backfilled, then constrained: an ADD COLUMN NOT NULL
        # without a default cannot succeed against existing rows.
        op.add_column(table, sa.Column("subject_id", sa.String(length=64), nullable=True))
        op.execute(f"UPDATE {table} SET subject_id = order_id WHERE subject_id IS NULL")  # noqa: S608
        op.alter_column(table, "subject_id", nullable=False)
        op.create_index(f"ix_{table}_subject_id", table, ["subject_id"], unique=False)

        # A situation investigation has no single order, so the column that
        # used to be the subject becomes optional.
        op.alter_column(
            table, "order_id", existing_type=sa.VARCHAR(length=32), nullable=True
        )

    op.create_index(
        "ix_investigations_subject_type", "investigations", ["subject_type"], unique=False
    )
    for table in ("proposals", "tickets"):
        op.add_column(
            table,
            sa.Column(
                "member_order_ids",
                postgresql.JSONB(astext_type=sa.Text()),
                nullable=True,
            ),
        )


def downgrade() -> None:
    """Reverse the migration.

    Destructive by necessity: a situation row has no `order_id` and cannot be
    represented in the old schema, so those rows are removed rather than
    silently given a fabricated order id.
    """
    for table in ("tickets", "proposals", "investigations"):
        op.execute(f"DELETE FROM {table} WHERE subject_type = 'situation'")  # noqa: S608

    for table in ("proposals", "tickets"):
        op.drop_column(table, "member_order_ids")
    op.drop_index("ix_investigations_subject_type", table_name="investigations")

    for table in _TABLES:
        op.alter_column(
            table, "order_id", existing_type=sa.VARCHAR(length=32), nullable=False
        )
        op.drop_index(f"ix_{table}_subject_id", table_name=table)
        op.drop_column(table, "subject_id")
        op.drop_column(table, "subject_type")
