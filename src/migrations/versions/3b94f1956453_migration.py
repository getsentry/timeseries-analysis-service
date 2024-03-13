"""Migration

Revision ID: 3b94f1956453
Revises: 82a6d42b94cf
Create Date: 2024-03-13 21:49:58.938941

"""
import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "3b94f1956453"
down_revision = "82a6d42b94cf"
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    with op.batch_alter_table("repositories", schema=None) as batch_op:
        batch_op.alter_column(
            "organization",
            existing_type=sa.INTEGER(),
            type_=sa.BigInteger(),
            existing_nullable=False,
        )
        batch_op.alter_column(
            "project", existing_type=sa.INTEGER(), type_=sa.BigInteger(), existing_nullable=False
        )

    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    with op.batch_alter_table("repositories", schema=None) as batch_op:
        batch_op.alter_column(
            "project", existing_type=sa.BigInteger(), type_=sa.INTEGER(), existing_nullable=False
        )
        batch_op.alter_column(
            "organization",
            existing_type=sa.BigInteger(),
            type_=sa.INTEGER(),
            existing_nullable=False,
        )

    # ### end Alembic commands ###
