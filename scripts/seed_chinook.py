"""
Loads the Chinook music store database into an empty PostgreSQL database.

Runs on startup in the hosted demo, so a fresh Railway database fills itself.
Does nothing if the tables are already there.

    python scripts/seed_chinook.py                # uses DATABASE_URL
    python scripts/seed_chinook.py path/to.sql
"""
import os
import sys

import psycopg2

DEFAULT_SQL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "chinook.sql")


def _db_url() -> str:
    url = os.getenv("DATABASE_URL", "")
    if url.startswith("postgres://"):
        url = "postgresql://" + url[len("postgres://"):]
    return url


def already_loaded(conn) -> bool:
    with conn.cursor() as cur:
        cur.execute("SELECT to_regclass('public.artist') IS NOT NULL")
        if not cur.fetchone()[0]:
            return False
        cur.execute("SELECT COUNT(*) FROM artist")
        return cur.fetchone()[0] > 0


def clean_script(sql: str) -> str:
    """
    The upstream file is written for psql: it drops and creates a database
    and switches to it. Hosted databases already exist, so skip those lines.
    """
    kept = []
    for line in sql.split("\n"):
        stripped = line.strip().upper()
        if line.strip().startswith("\\"):
            continue
        if stripped.startswith("DROP DATABASE") or stripped.startswith("CREATE DATABASE"):
            continue
        kept.append(line)
    return "\n".join(kept)


def seed(sql_path: str = DEFAULT_SQL) -> bool:
    """Returns True if it loaded data, False if the data was already there."""
    conn = psycopg2.connect(_db_url())
    try:
        if already_loaded(conn):
            print("Chinook already loaded")
            return False
        with open(sql_path, encoding="utf-8") as f:
            script = clean_script(f.read())
        with conn.cursor() as cur:
            cur.execute(script)
        conn.commit()
        print("Chinook loaded")
        return True
    finally:
        conn.close()


if __name__ == "__main__":
    seed(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_SQL)
