import mesop as me

from hello_redhat.src.components.conversation import conversation
from hello_redhat.src.components.header import header
from hello_redhat.src.components.page_scaffold import page_frame, page_scaffold
from hello_redhat.src.state.state import AppState


def conversation_page(app_state: AppState):
    """Conversation Page"""
    state = me.state(AppState)
    with page_scaffold():  # pylint: disable=not-context-manager
        with page_frame():
            with header('Conversation', 'chat'):
                pass
            conversation()


# def conversation_page(app_state: AppState):
#     state = me.state(AppState)
#     with page_scaffold():  # pylint: disable=not-context-manager
#         with page_frame():
#             # Add the image container with center alignment
#             with me.box(
#                     style=me.Style(
#                         display='flex',
#                         flex_direction='column',
#                         align_items='center',
#                         margin=me.Margin(bottom=20, top=20),
#                         width='100%',
#                     )
#             ):
#                 # Add the image
#                 me.image(
#                     src='https://picsum.photos/400/300',  # Replace with your image URL
#                     style=me.Style(
#                         max_width='400px',
#                         width='100%',
#                         height='auto',
#                         border_radius=8,
#                         box_shadow='0 4px 6px rgba(0, 0, 0, 0.1)',
#                     ),
#                 )
#                 # Add the caption text below the image
#                 me.text(
#                     'Your image caption here',  # Replace with your desired caption
#                     style=me.Style(
#                         margin=me.Margin(top=12),
#                         color=me.theme_var('on-surface-variant'),
#                         font_size='14px',
#                         text_align='center',
#                     ),
#                 )
#
#             # Original header
#             with header('Conversation', 'chat'):
#                 pass
#             conversation()