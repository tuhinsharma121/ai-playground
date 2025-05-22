from contextlib import AbstractAsyncContextManager

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.postgres import AsyncPostgresStore

from agent_redhat.src.constants import constants
from utils.pylogger import get_python_logger

logger = get_python_logger(log_level=constants.LOG_LEVEL)


def validate_postgres_config() -> None:
    """
    Validate that all required PostgreSQL configuration is present.
    Raises ValueError if any required configuration is missing.
    """
    required_vars = [
        "POSTGRES_USER",
        "POSTGRES_PASSWORD",
        "POSTGRES_HOST",
        "POSTGRES_PORT",
        "POSTGRES_DB",
    ]

    missing = [var for var in required_vars if not getattr(constants, var, None)]
    if missing:
        raise ValueError(
            f"Missing required PostgreSQL configuration: {', '.join(missing)}. "
            "These environment variables must be set to use PostgreSQL persistence."
        )


def get_postgres_connection_string() -> str:
    """Build and return the PostgreSQL connection string from settings."""
    if constants.POSTGRES_PASSWORD is None:
        raise ValueError("POSTGRES_PASSWORD is not set")
    connection_string = (
        f"postgresql://{constants.POSTGRES_USER}:"
        f"{constants.POSTGRES_PASSWORD.get_secret_value()}@"
        f"{constants.POSTGRES_HOST}:{constants.POSTGRES_PORT}/"
        f"{constants.POSTGRES_DB}"
    )
    logger.debug("Using PostgreSQL connection string: %s", connection_string)
    return connection_string


def get_postgres_saver() -> AbstractAsyncContextManager[AsyncPostgresSaver]:
    """Initialize and return a PostgreSQL saver instance."""
    validate_postgres_config()
    return AsyncPostgresSaver.from_conn_string(get_postgres_connection_string())


def get_postgres_store():
    """
    Get a PostgreSQL store instance.

    Returns an AsyncPostgresStore instance that needs to be used with async context manager
    pattern according to the documentation:

    async with AsyncPostgresStore.from_conn_string(conn_string) as store:
        await store.setup()  # Run migrations
        # Use store...
    """
    validate_postgres_config()
    connection_string = get_postgres_connection_string()
    return AsyncPostgresStore.from_conn_string(connection_string)