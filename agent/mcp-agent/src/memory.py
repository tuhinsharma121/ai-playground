from contextlib import AbstractAsyncContextManager
from contextlib import asynccontextmanager

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.store.memory import InMemoryStore

from constants import constants


def get_sqlite_saver() -> AbstractAsyncContextManager[AsyncSqliteSaver]:
    """Initialize and return a SQLite saver instance."""
    return AsyncSqliteSaver.from_conn_string(constants.SQLITE_DB_PATH)


class AsyncInMemoryStore:
    """Wrapper for InMemoryStore that provides an async context manager interface."""

    def __init__(self):
        self.store = InMemoryStore()

    async def __aenter__(self):
        return self.store

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        # No cleanup needed for InMemoryStore
        pass

    async def setup(self):
        # No-op method for compatibility with PostgresStore
        pass


@asynccontextmanager
async def get_sqlite_store():
    """Initialize and return a store instance for long-term memory.

    Note: SQLite-specific store isn't available in LangGraph,
    so we use InMemoryStore wrapped in an async context manager for compatibility.
    """
    store_manager = AsyncInMemoryStore()
    yield await store_manager.__aenter__()


def initialize_database() -> AbstractAsyncContextManager[
    AsyncSqliteSaver
    ]:
    """
    Initialize the appropriate database checkpointer based on configuration.
    Returns an initialized AsyncCheckpointer instance.
    """
    return get_sqlite_saver()


def initialize_store():
    """
    Initialize the appropriate store based on configuration.
    Returns an async context manager for the initialized store.
    """
    return get_sqlite_store()
