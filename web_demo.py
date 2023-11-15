import json
import torch
import streamlit as st

from evaluator import WenshuEvaluator, LawQAEvaluator
from retriever import Retriever


st.set_page_config(page_title="BUPT-LawLLM ⚖️")
sidebar_options = {"法律咨询问答 👩🏻‍💼": "feature_1", "法律文书补全 📃": "feature_2", "相关法条检索 🔎": "feature_3"}
st.sidebar.markdown("## BUPT-LawLLM ⚖️")
st.sidebar.markdown("我是 BUPT 智能系统实验室基于 ChatGLM3 等开源 LLM 微调并设计的法律领域 AI 助手。")
selected_option = st.sidebar.selectbox("", list(sidebar_options.keys()))

if 'case' not in st.session_state:
    st.session_state['case'] = ""
if 'judgeAccusation' not in st.session_state:
    st.session_state['judgeAccusation'] = ""
if 'judgeReason' not in st.session_state:
    st.session_state['judgeReason'] = ""


@st.cache_resource
def init_model():
    model = WenshuEvaluator(k=0, top_p=0.9, temperature=0.1, relevant_k=5, finetuned=True, source=None, verbose=True)
    retriever = Retriever(relevant_k=5, source="local") 
    return model, retriever

def clear_chat_history():
    if st.session_state.messages:
        del st.session_state.messages

def init_chat_history():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    for message in st.session_state.messages:
        avatar = "🧑🏻‍💻" if message["role"] == "user" else "⚖️"
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])

    return st.session_state.messages


def main():
    model, retriever = init_model()
    
    with st.chat_message("assistant", avatar="⚖️"):
        st.markdown(f"您好，我是 BUPT-LawLLM，很高兴为您服务。您可以在左侧边栏切换我的功能模式。如果我的生成有误，请以专业律师和国家出台的法条为基准。") 

    if sidebar_options[selected_option] == "feature_1":
        messages = init_chat_history()

        if prompt := st.chat_input("Shift + Enter 换行，Enter 发送"):
            with st.chat_message("user", avatar="🧑🏻‍💻"):
                st.markdown(prompt)

            messages.append({"role": "user", "content": prompt})
            print(f"[user] {prompt}", flush=True)
            with st.chat_message("assistant", avatar="👩🏻‍💼"):
                placeholder = st.empty()
                responses = [model.model_generate("你是一个专业律师，如下是一个法律咨询问题："+message["content"]+"\n请根据相关法条和你的知识进行回答。你的回答：") for message in messages]
                for response in responses:
                    placeholder.markdown(response)
                    if torch.backends.mps.is_available():
                        torch.mps.empty_cache()

            messages.append({"role": "assistant", "content": response})
            print(json.dumps(messages, ensure_ascii=False), flush=True)
            st.button("清空对话", on_click=clear_chat_history)

    elif sidebar_options[selected_option] == "feature_2":
        messages = init_chat_history()

        # if not st.session_state.case and not st.session_state.judgeAccusation and not st.session_state.judgeReason:
        #     with st.chat_message("assistant", avatar="📃"):
        #         st.markdown("文书的案件名称是什么？")

        if step := st.chat_input(f"Shift + Enter 换行，Enter 发送"):
            if (not st.session_state.case) and (not st.session_state.judgeAccusation) and (not st.session_state.judgeReason):
                st.session_state.case = step
                with st.chat_message("user", avatar="🧑🏻‍💻"):
                    st.markdown("已提供案件名称。")
                with st.chat_message("assistant", avatar="📃"):
                    st.markdown("好的，案件的指控部分说了什么？")

            elif (st.session_state.case) and (not st.session_state.judgeAccusation) and (not st.session_state.judgeReason):
                st.session_state.judgeAccusation = step
                with st.chat_message("user", avatar="🧑🏻‍💻"):
                    st.markdown("已提供指控部分。")
                with st.chat_message("assistant", avatar="📃"):
                    st.markdown("好的，那么法院给出的过程推理是怎样的？")
                # messages.append({"role": "assistant", "content": "好的，那么法院给出的过程推理是怎样的？"})

            elif (st.session_state.case) and (st.session_state.judgeAccusation) and (not st.session_state.judgeReason):
                st.session_state.judgeReason = step
                prompt = {"Case": st.session_state.case, "JudgeAccusation": st.session_state.judgeAccusation, "JudgeReason": st.session_state.judgeReason}
                messages.append({"role": "user", "content": prompt})
                print(f"[user] {prompt}", flush=True)

                with st.chat_message("assistant", avatar="📃"):
                    placeholder = st.empty()
                    responses = [model.run(message["content"]) for message in messages]
                    for response in responses:
                        placeholder.markdown(response)
                        if torch.backends.mps.is_available():
                            torch.mps.empty_cache()

                messages.append({"role": "assistant", "content": response})
                print(json.dumps(messages, ensure_ascii=False), flush=True)
                st.session_state.case, st.session_state.judgeAccusation, st.session_state.judgeReason = "", "", ""
                st.button("清空对话", on_click=clear_chat_history)

    elif sidebar_options[selected_option] == "feature_3":
        messages = init_chat_history()

        if prompt := st.chat_input("Shift + Enter 换行，Enter 发送"):
            with st.chat_message("user", avatar="🧑🏻‍💻"):
                st.markdown(prompt)

            messages.append({"role": "user", "content": prompt})
            print(f"[user] {prompt}", flush=True)
            with st.chat_message("assistant", avatar="🔎"):
                placeholder = st.empty()
                outputs = []
                for message in messages:
                    output = retriever.query(prompt)
                    outputs.append(output)
                
                for response in outputs:
                    placeholder.markdown(response)
                    if torch.backends.mps.is_available():
                        torch.mps.empty_cache()

            messages.append({"role": "assistant", "content": response})
            print(json.dumps(messages, ensure_ascii=False), flush=True)
            st.button("清空对话", on_click=clear_chat_history)


if __name__ == "__main__":
    main()
