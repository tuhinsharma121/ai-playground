import mesop as me

from ui.src.components.event_viewer import event_list
from ui.src.components.header import header
from ui.src.components.page_scaffold import page_frame, page_scaffold
from ui.src.state.agent_state import AgentState
from ui.src.state.state import AppState


def event_list_page(app_state: AppState):
    """Agents List Page"""
    state = me.state(AgentState)
    with page_scaffold():  # pylint: disable=not-context-manager
        with page_frame():
            with header('Event List', 'list'):
                pass
            event_list()
