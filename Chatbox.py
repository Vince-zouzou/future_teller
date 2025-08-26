import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import warnings
import streamlit as st
import pandas as pd
from resources.UIs.ChatboxUI import message_func,StreamlitUICallbackHandler  # 保留用于消息展示的函数
from backend.multi_agent_workflow import MultiAgentWorkflowSystem


#from agent import create_agent
warnings.filterwarnings("ignore")

def submit_query():
    st.session_state.responsed = False

def show_KG(data):
    from pyvis.network import Network

    # 中间显示知识图谱
    net = Network(notebook=False, height="750px", width="100%", directed=True)
    
    # Add nodes
    for node in data["knowledge_graph"]["nodes"]:
        net.add_node(node["id"], label=node["label"], title=node["type"])

    # Add edges
    for edge in data["knowledge_graph"]["edges"]:
        net.add_edge(edge["from"], edge["to"], label=edge["label"], title=edge["relation"], arrow_strikethrough=False)

    # Generate and save the HTML file
    net.write_html("knowledge_graph1.html")
    import streamlit.components.v1 as components
    with open("knowledge_graph1.html", "r", encoding="utf-8") as f:
        html_data = f.read()
        components.html(html_data, height=800)

def chatbox():
    dsk_key = "sk-d8ed661d7cbc4e86ba457abc3e794b90"
    cue = """
    - Answer questions directly or engage in casual conversation
    - Keep responses concise and helpful
    """
    
    # 初始化多Agent工作流系统 - 使用DeepSeek API


    initial  = "Hi, I am the AI Future Teller. How can I help you?"


    INITIAL_MESSAGE = [
        {
            "role": "assistant",
            "content": {'text':f"{initial}", "files":[]}}    ,
    ]



    if st.session_state.reset:
        st.session_state["data_messages"] = INITIAL_MESSAGE
        st.session_state["data_history"] = []
    
    if "data_messages" not in st.session_state.keys():
        st.session_state["data_messages"] = INITIAL_MESSAGE

    if "data_history" not in st.session_state:
        st.session_state["data_history"] = []
    
    if 'analysis_data' not in st.session_state:
        st.session_state.analysis_data = None

    if 'intent' not in st.session_state:
        st.session_state.intent = None

    if 'hist' not in st.session_state:
        st.session_state.hist = ' '

    with st.sidebar:
        conversation = st.container(key="cc", border=True, height=1000)
        inputbox = st.container(key="ii", border=False)
    # 根据是否扩展显示不同的布局
    cot,back = st.tabs(["🤖 Final report", "🔧 Backend"])
    st.session_state.back = back
    if st.session_state.analysis_data and st.session_state.intent not in (None,"normal"):

            # 显示分析结果 - 按照用户要求的布局
        data = st.session_state.analysis_data
        with cot:
            # 最下方显示时间线表格
            if 'timeline' in data :
                st.subheader("#### 📅 Timeline table")
                timeline_df = pd.DataFrame(data['timeline'])
                st.dataframe(timeline_df,hide_index=True)
                #st.table(timeline_df[['date', 'title', 'description']])

            show_KG(data)
            
            if 'response' in data:
                st.write("#### 📋 Analysis report")
                st.write(data['response'])
        
    messages_to_display = st.session_state["data_messages"].copy()
    with inputbox:
        if st.session_state.responsed:
            chat_input = st.chat_input(accept_file='multiple',on_submit=submit_query,disabled=False)
        else:
            chat_input = st.chat_input(accept_file='multiple',on_submit=submit_query,disabled=True)
            
        if prompt := chat_input:
            text = prompt.text
            files = prompt.files
            if len(text) > 500:
                st.error("Input is too long! Please limit your message to 500 characters.")
            else:
                st.session_state["data_messages"].append({"role": "user", "content": {'text':text,"file":files}})
                st.session_state["assistant_response_processed"] = False
            
            messages_to_display = st.session_state["data_messages"].copy()
            
    with conversation:
        for message in messages_to_display:
            message_func(
                message["content"]['text'],
                is_user=(message["role"] == "user"),
            )

    def append_message(content, role="assistant"):
        """Appends a message to the session state messages."""
        if content.strip():
            try:st.session_state["data_messages"].append({"role": role, "content": {'text':content.text,'file':content.file}})
            except: st.session_state["data_messages"].append({"role": role, "content": {'text':content,'file':None}})
    
    # 处理更新的信息
    with conversation:
        callback_handler = StreamlitUICallbackHandler("")
        if ( 
            "data_messages" in st.session_state
            and st.session_state["data_messages"][-1]["role"] == "user"
            and not st.session_state["assistant_response_processed"]
        ):
            user_input_content = st.session_state["data_messages"][-1]["content"]

            if isinstance(user_input_content['text'], str):
                # Start loading animation with initial step
                callback_handler.start_loading_message("Analysing user intention")
                final_input = user_input_content['text']
                
                
                ## to add historical information
                history_string = "Chat History:"
                if len(st.session_state["data_messages"]) > 6:
                    for item in st.session_state["data_messages"][-1:-7]:
                        history_string = history_string + "(" + item['role']+":"+item["content"]['text'] + ");"
                    cue = history_string + cue
                else:                    
                    for item in st.session_state["data_messages"]:
                        history_string = history_string + "(" + item['role']+":"+item["content"]['text'] + ");"
                    cue = history_string + cue
                cue += str(st.session_state.hist)
                multi_agent_system = MultiAgentWorkflowSystem(
                            api_key= dsk_key,  # DeepSeek API key
                            api_version="2023-05-15",
                            azure_endpoint="https://hkust.azure-api.net",
                            cue = cue,
                            use_deepseek=True )
                # 所有查询都通过multi_agent_workflow处理，由其内部进行意图识别和路由
                callback_handler.start_loading_message("Processing with multi-agent workflow")
                
                # 使用多Agent工作流系统处理所有查询
                def update_status(message):
                    callback_handler.start_loading_message(message)
            
                result = multi_agent_system.process_query(final_input, status_callback=update_status)
                
                if st.session_state.intent == 'normal':
                    pass
                else: st.session_state.analysis_data = result
                

                # 获取格式化后的响应内容
                response = result.get('response', str(result))
                
                if st.session_state.intent == "normal":
                    words = [str(i) for i in response ] 
                else:
                    words = "Finished Anlysis, check on the left"
                    st.session_state.hist = response
                
                for w in words:
                    callback_handler.on_llm_new_token(w, run_id="fixed_run_id")
                    #time.sleep(0.001)  # 可选：模拟思考延迟
                # 输出完成后调用on_llm_end
                callback_handler.on_llm_end(response={"final": "done"}, run_id="fixed_run_id")
                # 此时 callback_handler.final_message 就是固定输出的完整文本
                assistant_message = callback_handler.final_message
                append_message(assistant_message)
                #append_message(response)
                st.session_state["assistant_response_processed"] = True
                st.session_state.responsed = True
                st.rerun()
                

