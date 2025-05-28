import mesop as me

from hello_redhat.src.components.event_viewer import event_list
from hello_redhat.src.components.header import header
from hello_redhat.src.components.page_scaffold import page_frame, page_scaffold
from hello_redhat.src.state.agent_state import AgentState
from hello_redhat.src.state.state import AppState


def event_list_page(app_state: AppState):
    """Agents List Page"""
    state = me.state(AgentState)
    with page_scaffold():  # pylint: disable=not-context-manager
        with page_frame():
            with header('Event List', 'list'):
                pass
            event_list()
