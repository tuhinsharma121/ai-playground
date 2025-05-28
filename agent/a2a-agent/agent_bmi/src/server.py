import os

from agent_bmi.src.agent import BMIAgent
from common.server import A2AServer
from common.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
    MissingAPIKeyError,
)
from common.utils.push_notification_auth import PushNotificationSenderAuth
from agent_bmi.src.task_manager import AgentTaskManager
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
            id='calcualte_bmi',
            name='Calculate BMI Tool',
            description= 'Helps you with calculating your BMI (Body Mass Index) given height in centimeters and weight in kilograms.',
            tags=['calculate bmi'],
            examples=['What is the BMI of 175 cm and 70.5 kg ?'],
        )

        host = os.getenv("HOST", "0.0.0.0")
        port = int(os.getenv("PORT", 2001))
        agent_card = AgentCard(
            name='BMI Agent',
            description= \
                        """
                           - Use this agent when the user asks about calculating BMI.
                           - This tool requires a height in cms and weight in kgs for a person. If not provided use other agents to infer it.
                           - You can use information from the other agents if height and weight are not provided.
                        """
            ,
            url=f'http://{host}:{port}/',
            version='1.0.0',
            defaultInputModes=BMIAgent.SUPPORTED_CONTENT_TYPES,
            defaultOutputModes=BMIAgent.SUPPORTED_CONTENT_TYPES,
            capabilities=capabilities,
            skills=[skill],
        )

        notification_sender_auth = PushNotificationSenderAuth()
        notification_sender_auth.generate_jwk()
        server = A2AServer(
            agent_card=agent_card,
            task_manager=AgentTaskManager(
                agent=BMIAgent(),
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
