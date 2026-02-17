import os
import urllib.parse
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# URL for the database connection.
# Ensure you have a running PostgreSQL instance and a database named 'frs_db'.
# You might need to change 'user:password' to your actual credentials.

DB_USER = os.getenv("DB_USER", "murali")
DB_PASSWORD = os.getenv("DB_PASSWORD", "Murali@2004")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_NAME = os.getenv("DB_NAME", "frs_db")

encoded_password = urllib.parse.quote_plus(DB_PASSWORD)
SQLALCHEMY_DATABASE_URL = f"postgresql://{DB_USER}:{encoded_password}@{DB_HOST}/{DB_NAME}"

engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
