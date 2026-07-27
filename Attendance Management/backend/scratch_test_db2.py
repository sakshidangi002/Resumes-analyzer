import os
import sys
from dotenv import load_dotenv
from sqlalchemy import create_engine

load_dotenv()

postgres_host = os.environ.get("POSTGRES_HOST")
postgres_port = os.environ.get("POSTGRES_PORT", 5432)
postgres_user = os.environ.get("POSTGRES_USER")
postgres_password = os.environ.get("POSTGRES_PASSWORD")
postgres_db = os.environ.get("POSTGRES_DB")

database_url = (
    f"postgresql+psycopg2://{postgres_user}:{postgres_password}"
    f"@{postgres_host}:{postgres_port}/{postgres_db}"
)

print(f"Connecting with SQLAlchemy to: {database_url}")
try:
    engine = create_engine(database_url, pool_pre_ping=True, connect_args={"connect_timeout": 5})
    with engine.connect() as conn:
        print("SQLAlchemy Connection successful!")
except Exception as e:
    print(f"SQLAlchemy Connection failed: {e}")
