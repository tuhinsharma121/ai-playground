import mesop as me
import os
from fastapi import APIRouter, FastAPI
from fastapi.middleware.wsgi import WSGIMiddleware

from pylogger import get_uvicorn_log_config
from hello_redhat.src.components.api_key_dialog import api_key_dialog
from hello_redhat.src.components.page_scaffold import page_scaffold
from hello_redhat.src.pages.agent_list import agent_list_page
from hello_redhat.src.pages.conversation import conversation_page
from hello_redhat.src.pages.event_list import event_list_page
from hello_redhat.src.pages.home import home_page_content
from hello_redhat.src.pages.settings import settings_page_content
from hello_redhat.src.pages.task_list import task_list_page
from hello_redhat.src.service.server.server import ConversationServer
from hello_redhat.src.state import host_agent_service
from hello_redhat.src.state.state import AppState


def on_load(e: me.LoadEvent):  # pylint: disable=unused-argument
    """On load event"""
    state = me.state(AppState)
    me.set_theme_mode(state.theme_mode)
    if 'conversation_id' in me.query_params:
        state.current_conversation_id = me.query_params['conversation_id']
    else:
        state.current_conversation_id = ''

    # check if the API key is set in the environment
    # and if the user is using Vertex AI
    uses_vertex_ai = (
            os.getenv('GOOGLE_GENAI_USE_VERTEXAI', '').upper() == 'TRUE'
    )
    api_key = os.getenv('GOOGLE_API_KEY', '')

    if uses_vertex_ai:
        state.uses_vertex_ai = True
    elif api_key:
        state.api_key = api_key
    else:
        # Show the API key dialog if both are not set
        state.api_key_dialog_open = True


# Policy to allow the lit custom element to load
security_policy = me.SecurityPolicy(
    allowed_script_srcs=[
        'https://cdn.jsdelivr.net',
    ]
)


@me.page(
    path='/',
    title='Chat',
    on_load=on_load,
    security_policy=security_policy,
)
def home_page():
    """Main Page"""
    state = me.state(AppState)
    # Show API key dialog if needed
    api_key_dialog()
    with page_scaffold():  # pylint: disable=not-context-manager
        home_page_content(state)


@me.page(
    path='/agents',
    title='Agents',
    on_load=on_load,
    security_policy=security_policy,
)
def another_page():
    """Another Page"""
    api_key_dialog()
    agent_list_page(me.state(AppState))


@me.page(
    path='/conversation',
    title='Conversation',
    on_load=on_load,
    security_policy=security_policy,
)
def chat_page():
    """Conversation Page."""
    api_key_dialog()
    conversation_page(me.state(AppState))


@me.page(
    path='/event_list',
    title='Event List',
    on_load=on_load,
    security_policy=security_policy,
)
def event_page():
    """Event List Page."""
    api_key_dialog()
    event_list_page(me.state(AppState))


@me.page(
    path='/settings',
    title='Settings',
    on_load=on_load,
    security_policy=security_policy,
)
def settings_page():
    """Settings Page."""
    api_key_dialog()
    settings_page_content()


@me.page(
    path='/task_list',
    title='Task List',
    on_load=on_load,
    security_policy=security_policy,
)
def task_page():
    """Task List Page."""
    api_key_dialog()
    task_list_page(me.state(AppState))


# Setup the server global objects
app = FastAPI()
from fastapi.staticfiles import StaticFiles
app.mount("/static", StaticFiles(directory="hello_redhat/src/static"), name="static")
router = APIRouter()
agent_server = ConversationServer(router)
app.include_router(router)

app.mount(
    '/',
    WSGIMiddleware(
        me.create_wsgi_app(
            debug_mode=os.environ.get('DEBUG_MODE', '') == 'true'
        )
    ),
)

if __name__ == '__main__':
    import uvicorn

    # Setup the connection details, these should be set in the environment
    host = os.getenv('A2A_UI_HOST', '0.0.0.0')
    port = int(os.getenv('A2A_UI_PORT', '8501'))

    uvicorn.run(
        app='app:app',
        host=host,
        port=port,
        reload=True,
        reload_includes=['*.py', '*.js'],
        timeout_graceful_shutdown=0,
        log_config=get_uvicorn_log_config(),
        log_level='critical',
    )
