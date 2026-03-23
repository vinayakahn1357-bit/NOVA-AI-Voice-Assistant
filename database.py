"""
database.py — SQLAlchemy ORM Layer for NOVA (Phase 6)
Provides PostgreSQL/SQLite database models, engine creation, and session factory.
Falls back to SQLite when DATABASE_URL is not set or points to SQLite.
"""

import os
from datetime import datetime, timezone

from sqlalchemy import (
    create_engine, Column, Integer, String, Float, Text, DateTime,
    ForeignKey, UniqueConstraint, Index, event,
)
from sqlalchemy.orm import declarative_base, sessionmaker, relationship

from utils.logger import get_logger

log = get_logger("database")

Base = declarative_base()


# ─── ORM Models ───────────────────────────────────────────────────────────────

class DBUser(Base):
    __tablename__ = "users"

    id = Column(String(64), primary_key=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    name = Column(String(255), nullable=True)
    password_hash = Column(Text, nullable=True)
    salt = Column(String(64), nullable=True)
    provider = Column(String(32), default="email")  # email | google
    role = Column(String(16), default="user")        # user | admin
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    # Relationships
    sessions = relationship("DBSession", back_populates="user", cascade="all, delete-orphan")
    memory_facts = relationship("DBMemoryFact", back_populates="user", cascade="all, delete-orphan")
    preferences = relationship("DBUserPreference", back_populates="user", cascade="all, delete-orphan")
    interests = relationship("DBUserInterest", back_populates="user", cascade="all, delete-orphan")


class DBSession(Base):
    __tablename__ = "chat_sessions"

    id = Column(String(64), primary_key=True)
    user_id = Column(String(64), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    user = relationship("DBUser", back_populates="sessions")
    messages = relationship("DBMessage", back_populates="session", cascade="all, delete-orphan",
                            order_by="DBMessage.turn")


class DBMessage(Base):
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(64), ForeignKey("chat_sessions.id", ondelete="CASCADE"), nullable=False, index=True)
    turn = Column(Integer, nullable=False)
    role = Column(String(16), nullable=False)  # user | nova | assistant
    content = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    session = relationship("DBSession", back_populates="messages")

    __table_args__ = (
        UniqueConstraint("session_id", "turn", name="uq_session_turn"),
        Index("ix_message_session_turn", "session_id", "turn"),
    )


class DBMemoryFact(Base):
    __tablename__ = "memory_facts"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(64), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    fact = Column(Text, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    user = relationship("DBUser", back_populates="memory_facts")

    __table_args__ = (
        UniqueConstraint("user_id", "fact", name="uq_user_fact"),
    )


class DBUserPreference(Base):
    __tablename__ = "user_preferences"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(64), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    key = Column(String(128), nullable=False)
    value = Column(Text, nullable=False)

    user = relationship("DBUser", back_populates="preferences")

    __table_args__ = (
        UniqueConstraint("user_id", "key", name="uq_user_pref_key"),
    )


class DBUserInterest(Base):
    __tablename__ = "user_interests"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(64), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    topic = Column(String(256), nullable=False)
    count = Column(Integer, default=1)

    user = relationship("DBUser", back_populates="interests")

    __table_args__ = (
        UniqueConstraint("user_id", "topic", name="uq_user_interest"),
    )


class DBDailySummary(Base):
    __tablename__ = "daily_summaries"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(64), nullable=False, index=True)
    day = Column(String(10), nullable=False)
    summary = Column(Text, nullable=False)

    __table_args__ = (
        UniqueConstraint("user_id", "day", name="uq_user_day"),
    )


class DBMeta(Base):
    __tablename__ = "nova_meta"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(64), nullable=False, index=True)
    key = Column(String(128), nullable=False)
    value = Column(Text, nullable=False)

    __table_args__ = (
        UniqueConstraint("user_id", "key", name="uq_user_meta_key"),
    )


# ─── Engine & Session Factory ────────────────────────────────────────────────

_engine = None
_SessionFactory = None


def _get_database_url() -> str:
    """Resolve DATABASE_URL with smart defaults."""
    from config import BASE_DIR, IS_VERCEL

    url = os.getenv("DATABASE_URL", "")

    if url:
        # Heroku-style fix: postgres:// → postgresql://
        if url.startswith("postgres://"):
            url = url.replace("postgres://", "postgresql://", 1)
        return url

    # Default: SQLite (backward compat)
    if IS_VERCEL:
        return "sqlite:////tmp/nova_memory.db"
    return f"sqlite:///{os.path.join(BASE_DIR, 'nova_memory.db')}"


def is_postgres() -> bool:
    """Check if the configured database is PostgreSQL."""
    url = _get_database_url()
    return url.startswith("postgresql://") or url.startswith("postgresql+")


def get_engine():
    """Get or create the SQLAlchemy engine."""
    global _engine
    if _engine is None:
        url = _get_database_url()
        is_pg = url.startswith("postgresql")

        engine_kwargs = {
            "echo": False,
            "future": True,
        }

        if is_pg:
            engine_kwargs["pool_size"] = 5
            engine_kwargs["max_overflow"] = 10
            engine_kwargs["pool_pre_ping"] = True
            engine_kwargs["pool_recycle"] = 300
        else:
            # SQLite — enable WAL mode via event
            engine_kwargs["connect_args"] = {"check_same_thread": False, "timeout": 30}

        _engine = create_engine(url, **engine_kwargs)

        if not is_pg:
            @event.listens_for(_engine, "connect")
            def _set_wal(dbapi_conn, connection_record):
                cursor = dbapi_conn.cursor()
                cursor.execute("PRAGMA journal_mode=WAL")
                cursor.execute("PRAGMA synchronous=NORMAL")
                cursor.execute("PRAGMA foreign_keys=ON")
                cursor.close()

        db_type = "PostgreSQL" if is_pg else "SQLite"
        log.info("Database engine: %s", db_type)

    return _engine


def get_db_session():
    """Get a new database session."""
    global _SessionFactory
    if _SessionFactory is None:
        _SessionFactory = sessionmaker(bind=get_engine(), expire_on_commit=False)
    return _SessionFactory()


def init_db():
    """Create all tables if they don't exist."""
    engine = get_engine()
    Base.metadata.create_all(engine)
    db_type = "PostgreSQL" if is_postgres() else "SQLite"
    log.info("Database tables initialized (%s)", db_type)
