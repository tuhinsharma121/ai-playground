import os

from agent_email.src.agent import EmailAgent
from agent_email.src.common.server import A2AServer
from agent_email.src.common.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
    MissingAPIKeyError,
)
from agent_email.src.common.utils.push_notification_auth import PushNotificationSenderAuth
from agent_email.src.task_manager import AgentTaskManager
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
            id='send_email',
            name='Send Email Tool',
            description='Helps you send emails with recipients, subject, and message content.',
            tags=['send email', 'email communication'],
            examples=[
                'Send an email to john@example.com with subject "Meeting Update" and message "The meeting is postponed to tomorrow at 2 PM."'],
        )

        host = os.getenv("HOST", "0.0.0.0")
        port = int(os.getenv("PORT", 10002))
        agent_card = AgentCard(
            name='Email Agent',
            description="""
            - Use this agent when the user asks about sending emails.
            - This tool requires recipient subject, and message content. If not provided use other agents to infer it.
            - You can use information from other agents if email details are not provided.
            - Supports only single recipient.
            """
            ,
            url=f'http://{host}:{port}/',
            version='1.0.0',
            defaultInputModes=EmailAgent.SUPPORTED_CONTENT_TYPES,
            defaultOutputModes=EmailAgent.SUPPORTED_CONTENT_TYPES,
            capabilities=capabilities,
            skills=[skill],
        )

        notification_sender_auth = PushNotificationSenderAuth()
        notification_sender_auth.generate_jwk()
        server = A2AServer(
            agent_card=agent_card,
            task_manager=AgentTaskManager(
                agent=EmailAgent(),
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
