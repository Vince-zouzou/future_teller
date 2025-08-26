import html
import re
import os
import streamlit as st
from langchain.callbacks.base import BaseCallbackHandler

import base64


USER_MESSAGE_STYLE = """
<div style="display:flex; align-items:flex-start; justify-content:flex-end; margin:0; padding:0; margin-bottom:10px;">
    <div style="background:{message_bg_color}; color:white; border-radius:20px; padding:10px; margin-right:5px; max-width:75%; font-size:18px; margin:0; line-height:1.2; word-wrap:break-word;">
        {message_text}
    </div>
    <div style="width:40px; height:40px; margin:0; font-size:30px; line-height:40px; text-align:center;">🤔</div>
</div>
"""

BOT_MESSAGE_STYLE = """
<div style="display:flex; align-items:flex-start; justify-content:flex-start; margin:0; padding:0; margin-bottom:10px;">
    <div style="width:40px; height:40px; margin:0; margin-right:5px; margin-top:5px;font-size:30px; line-height:40px; text-align:center;">
        😄
    </div>
    <div style="background:{message_bg_color}; color:white; border-radius:20px; padding:10px; margin-left:5px; max-width:75%; font-size:18px; margin:0; line-height:1.2; word-wrap:break-word;">
        {message_text}
    </div>
</div>
"""

def format_message(text):
    """
    This function is used to format the messages in the chatbot UI.

    Parameters:
    text (str): The text to be formatted.
    """
    text_blocks = re.split(r"```[\s\S]*?```", text)
    code_blocks = re.findall(r"```([\s\S]*?)```", text)

    text_blocks = [html.escape(block) for block in text_blocks]

    formatted_text = ""
    for i in range(len(text_blocks)):
        formatted_text += text_blocks[i].replace("\n", "<br>")
        if i < len(code_blocks):
            formatted_text += f'<pre style="white-space: pre-wrap; word-wrap: break-word;"><code>{html.escape(code_blocks[i])}</code></pre>'

    return formatted_text


def message_func(text, is_user=False, model="gpt"):
    """
    This function displays messages in the chatbot UI, ensuring proper alignment and avatar positioning.

    Args:
    text (str): The text to be displayed.
    is_user (bool, optional): Whether the message is from the user or not. Defaults to False.
    model (str, optional): The model used to generate the message. Defaults to "gpt".

    Returns:
    None
    """
    message_bg_color = (
        "linear-gradient(135deg, #00B2FF 0%, #006AFF 100%)" if is_user else "#71797E"
    )
    message_text = html.escape(text.strip()).replace('\n', '<br>')

    if message_text:  # Check if message_text is not empty
        if is_user:
            container_html = USER_MESSAGE_STYLE.format(message_bg_color=message_bg_color, message_text=message_text)
        else:
            container_html = BOT_MESSAGE_STYLE.format(message_bg_color=message_bg_color, message_text=message_text)
        st.write(container_html, unsafe_allow_html=True)

class StreamlitUICallbackHandler(BaseCallbackHandler):
    def __init__(self, model):
        """
        类的初始化方法。

        Args:
            model (object): 使用的模型对象。

        Attributes:
            token_buffer (list): 用于存储标记的缓冲区。
            placeholder (st.empty): 一个空的Streamlit占位符，用于动态显示内容。
            has_streaming_ended (bool): 一个布尔值，表示流是否已经结束。
            has_streaming_started (bool): 一个布尔值，表示流是否已经开始。
            final_message (str): 流结束时的最终消息。
            model (object): 传入的模型对象。
        """
        self.token_buffer = []
        self.placeholder = st.empty()
        self.has_streaming_ended = False
        self.has_streaming_started = False
        self.final_message = ""
        self.model = model

    def start_loading_message(self, step_message="Analysing user intention"):
        loading_message_content = self._get_bot_message_container(step_message)
        self.placeholder.markdown(loading_message_content, unsafe_allow_html=True)

    def on_llm_new_token(self, token, run_id, parent_run_id=None, **kwargs):
        if not self.has_streaming_started:
            self.has_streaming_started = True
        self.token_buffer.append(str(token))
        self.final_message = "".join(self.token_buffer)

    def on_llm_end(self, response, run_id, parent_run_id=None, **kwargs):
        self.has_streaming_ended = True
        self.has_streaming_started = False
        complete_message = "".join(self.token_buffer)
        container_content = self._get_bot_message_container(complete_message)
        self.placeholder.markdown(container_content, unsafe_allow_html=True)
        self.token_buffer = []

    def _get_bot_message_container(self, text):
        """Generate the bot's message container style for the given text."""
        formatted_text = format_message(text.strip())
        if not formatted_text:
            formatted_text = "Thinking..."
        container_content = f"""
    <div style="display:flex; flex-direction:flex-start; align-items:center; justify-content:flex-start; margin:0; padding:0;">
        <div style="width:40px; height:40px; margin:0; margin-right:5px; margin-top:5px;font-size:30px; line-height:40px; text-align:center;">
           😄
        </div>
        <div style="background:#71797E; color:white; border-radius:20px; padding:10px; margin-top:5px; max-width:75%; font-size:18px; line-height:1.2; word-wrap:break-word;">
            {formatted_text}
        </div>
    </div>
    """
        return container_content

