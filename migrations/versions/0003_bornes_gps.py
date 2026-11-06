"""bornes GPS — points de repère nommés placés sur la carte.

Revision ID: 0003_bornes_gps
Revises: 0002_audit_entries
Create Date: 2026-10-28
"""

from typing import Sequence, Union

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "0003_bornes_gps"
down_revision: Union[str, None] = "0002_audit_entries"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "bornes_gps",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("lat", sa.Float(), nullable=False),
        sa.Column("lng", sa.Float(), nullable=False),
        sa.Column("created_at", sa.TIMESTAMP(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.TIMESTAMP(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.PrimaryKeyConstraint("id", name="bornes_gps_pkey"),
    )
    op.create_index("bornes_gps_name_idx", "bornes_gps", ["name"], unique=False)


def downgrade() -> None:
    op.drop_index("bornes_gps_name_idx", table_name="bornes_gps")
    op.drop_table("bornes_gps")
