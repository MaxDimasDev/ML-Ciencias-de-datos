import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base


DATABASE_URL = os.getenv(
    "DATABASE_URL",
    # Default to local sqlite for easy local dev; use Postgres in cloud
    "sqlite:///./local.db",
)


# For SQLAlchemy 2.0 style
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()