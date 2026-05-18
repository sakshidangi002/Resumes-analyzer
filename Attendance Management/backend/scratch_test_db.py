import os
import sys
from dotenv import load_dotenv
import psycopg2

load_dotenv()

host = os.environ.get("POSTGRES_HOST")
port = os.environ.get("POSTGRES_PORT", 5432)
user = os.environ.get("POSTGRES_USER")
password = os.environ.get("POSTGRES_PASSWORD")
dbname = os.environ.get("POSTGRES_DB")

print(f"Connecting to {host}:{port} as {user} to db {dbname}")
try:
    conn = psycopg2.connect(
        host=host,
        port=port,
        user=user,
        password=password,
        dbname=dbname,
        connect_timeout=5
    )
    print("Connection successful!")
    conn.close()
except Exception as e:
    print(f"Connection failed: {e}")
