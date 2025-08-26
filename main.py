from Chatbox import chatbox
import streamlit as st
import pandas as pd
from config import paths
st.set_page_config(page_title="Forcaster", page_icon="💬", layout="wide")
st.session_state.expand = False 


gradient_text_html = f"""
<div style="display: flex; align-items: center; justify-content: left; font-size: 30px;">
<!-- 移除图片，替换为Emoji -->
<span style="font-size: 35px; margin-right: 5px;">👑</span>
<!-- 渐变字体 Pro -->
<span style="font-weight: bold; 
            background: -webkit-linear-gradient(left, red, orange);
            background: linear-gradient(to right, red, orange);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-left: 0px;">
    The AI Future Teller
</span>
</div>
"""

st.sidebar.markdown(gradient_text_html, unsafe_allow_html=True)
st.session_state.reset = st.sidebar.button("Reset Message",width='stretch',type="primary")

st.session_state.responsed = True
with open(paths['style'], "r") as styles_file:
        styles_content = styles_file.read()
st.write(styles_content, unsafe_allow_html=True)
    


chatbox()

