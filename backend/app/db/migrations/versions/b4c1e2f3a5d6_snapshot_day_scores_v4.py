"""snapshot-day scores (v4): priority rank, components, arrival estimate

Revision ID: b4c1e2f3a5d6
Revises: 7ef6062d3e98
Create Date: 2026-10-04

Purely additive and every column is nullable, so the previous release keeps
working against the migrated schema while a deploy is in progress. The values
are loaded by `app.services.model_sync`, which the API runs at start-up.
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "b4c1e2f3a5d6"
down_revision = "7ef6062d3e98"
branch_labels = None
depends_on = None

_COLUMNS = [
    ("priority_rank", sa.Integer()),
    ("km_score", sa.Float()),
    ("hazard_score", sa.Float()),
    ("lambdamart_score", sa.Float()),
    ("eta_p10", sa.Date()),
    ("eta_p50", sa.Date()),
    ("eta_p90", sa.Date()),
    ("snapshot_features", postgresql.JSONB(astext_type=sa.Text())),
]


def upgrade() -> None:
    for name, type_ in _COLUMNS:
        op.add_column("snapshot_orders", sa.Column(name, type_, nullable=True))


def downgrade() -> None:
    for name, _ in reversed(_COLUMNS):
        op.drop_column("snapshot_orders", name)
