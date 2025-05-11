import mesop as me

from ui.src.components.conversation import conversation
from ui.src.components.header import header
from ui.src.components.page_scaffold import page_frame, page_scaffold
from ui.src.state.state import AppState


def conversation_page(app_state: AppState):
    """Conversation Page"""
    state = me.state(AppState)
    with page_scaffold():  # pylint: disable=not-context-manager
        with page_frame():
            with header('Conversation', 'chat'):
                pass
            conversation()
