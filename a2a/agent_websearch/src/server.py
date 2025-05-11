import logging
import os

from agent_websearch.src.agent import WebsearchAgent
from agent_websearch.src.task_manager import AgentTaskManager
from common.server import A2AServer
from common.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
    MissingAPIKeyError,
)
from common.utils.push_notification_auth import PushNotificationSenderAuth
from pylogger import get_uvicorn_log_config, get_python_logger

logger = get_python_logger()


def main():
    """Starts the Websearch Agent server."""
    try:
        if not os.getenv('GOOGLE_API_KEY'):
            raise MissingAPIKeyError(
                'GOOGLE_API_KEY environment variable not set.'
            )

        capabilities = AgentCapabilities(streaming=True, pushNotifications=True)
        skill = AgentSkill(
            id='search_web',
            name='Search Web Tool',
            description= 'Helps you with searching the web.',
            tags=['sarch web', 'tavily'],
            examples=['What is the current stock price of Google ?'],
        )

        host = os.getenv("HOST", "0.0.0.0")
        port = int(os.getenv("PORT", 10000))
        agent_card = AgentCard(
            name='Websearch Agent',
            description=\
            """
           - You can use this agent to get more information about a topic.
           - Use this when the user question cannot be answered by any other agents.
           - Call this when the topic is outside the coverage of the other agents.
           - You can use information from the other agents if the topic is not covered.
            """,
            url=f'http://{host}:{port}/',
            version='1.0.0',
            defaultInputModes=WebsearchAgent.SUPPORTED_CONTENT_TYPES,
            defaultOutputModes=WebsearchAgent.SUPPORTED_CONTENT_TYPES,
            capabilities=capabilities,
            skills=[skill],
        )

        notification_sender_auth = PushNotificationSenderAuth()
        notification_sender_auth.generate_jwk()
        server = A2AServer(
            agent_card=agent_card,
            task_manager=AgentTaskManager(
                agent=WebsearchAgent(),
                notification_sender_auth=notification_sender_auth,
            ),
            host="0.0.0.0",
            port=port,
            log_config=get_uvicorn_log_config()
        )

        server.app.add_route(
            '/.well-known/jwks.json',
            notification_sender_auth.handle_jwks_endpoint,
            methods=['GET'],
        )

        logger.info(f'Starting server on {host}:{port}')
        server.start()
    except MissingAPIKeyError as e:
        logger.error(f'Error: {e}')
        exit(1)
    except Exception as e:
        logger.error(f'An error occurred during server startup: {e}')
        exit(1)


if __name__ == '__main__':
    main()
