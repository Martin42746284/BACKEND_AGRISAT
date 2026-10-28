"""audit_entries — journal des actions utilisateurs.

Cette révision reconstruit la migration appliquée en production (la version
`0002_audit_entries` est déjà présente dans la table `alembic_version`). Elle n'est
donc pas rejouée sur la base existante, mais permet de garder un graphe Alembic
complet pour les environnements recréés depuis la baseline.

Revision ID: 0002_audit_entries
Revises: 0001_baseline
Create Date: 2026-09-28
"""

from typing import Sequence, Union

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "0002_audit_entries"
down_revision: Union[str, None] = "0001_baseline"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "audit_entries",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("user_name", sa.Text(), nullable=False),
        sa.Column("action", sa.Text(), nullable=False),
        sa.Column("object", sa.Text(), nullable=False),
        sa.Column("created_at", sa.TIMESTAMP(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.PrimaryKeyConstraint("id", name="audit_entries_pkey"),
    )
    op.create_index("audit_entries_created_at_idx", "audit_entries", ["created_at"], unique=False)


def downgrade() -> None:
    op.drop_index("audit_entries_created_at_idx", table_name="audit_entries")
    op.drop_table("audit_entries")
