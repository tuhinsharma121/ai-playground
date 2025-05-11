from ui.src.components.header import header
from ui.src.components.page_scaffold import page_frame, page_scaffold
from ui.src.components.task_card import task_card
from ui.src.state.state import AppState


def task_list_page(app_state: AppState):
    """Task List Page"""
    with page_scaffold():  # pylint: disable=not-context-manager
        with page_frame():
            with header('Task List', 'task'):
                pass
            task_card(app_state.task_list)
