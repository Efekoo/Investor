from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from crypto_bot.database.models import Base


class Database:
    def __init__(self, db_url: str) -> None:
        if db_url.startswith("sqlite:///"):
            sqlite_path = db_url.replace("sqlite:///", "", 1)
            Path(sqlite_path).parent.mkdir(parents=True, exist_ok=True)
        self.engine = create_engine(db_url, future=True)
        self._session = sessionmaker(bind=self.engine, autoflush=False, autocommit=False)

    def create_tables(self) -> None:
        Base.metadata.create_all(self.engine)

    @contextmanager
    def session_scope(self) -> Iterator[Session]:
        session = self._session()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
