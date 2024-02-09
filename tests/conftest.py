import pytest

from seer.db import Session, db


@pytest.fixture(autouse=True)
def manage_db():
    # Forces the initialization of the database
    from seer.app import app

    with app.app_context():
        Session.configure(bind=db.engine)
        db.metadata.create_all(bind=db.engine)
    try:
        yield
    finally:
        with app.app_context():
            db.metadata.drop_all(bind=db.engine)