import streamlit as st
import os
from dotenv import load_dotenv
from groq import Groq
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from llm import chat

load_dotenv()

# chunks = vs.get_text_chunks()
# vectorstore = vs.get_vectorstore(chunks)
# conversation_chain = vs.get_conversation_chain(vectorstore)

st.title("InnovaTech Solutions Chat")
question_audio = st.audio_input("Pregunta algo")
selected_model = st.selectbox("Modelo", [
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "meta-llama/llama-4-maverick-17b-128e-instruct",
    "deepseek-r1-distill-llama-70b",
    "mistral-saba-24b",
    "gemma2-9b-it"

])
question_text = st.chat_input("O tambien puedes escribir tu pregunta")

if not 'chat_history' in st.session_state:
    st.session_state.chat_history = []
if not 'conversation' in st.session_state:
    st.session_state.conversation = ""

user_input = ""
if question_audio:
    question_file = (question_audio.name, question_audio.getvalue())

    client = Groq()
    # filename = os.path.dirname(__file__) + "/audio.m4a"

    transcription = client.audio.transcriptions.create(
      file=question_file,
      model="whisper-large-v3",
      response_format="verbose_json"
    )
    user_input = transcription.text

elif question_text:
    user_input = question_text
    

if user_input:
    
    with st.spinner("Procesando..."):
        response, messages = chat(
            user_prompt=user_input, 
            selected_model=selected_model, 
            chat_history=st.session_state.chat_history
            )
        st.session_state.chat_history.append(HumanMessage(content=user_input))
        st.session_state.chat_history.append(AIMessage(content=response))
    st.subheader("Historial de Chat:")
    for msg in messages[::-1]:
        if isinstance(msg, HumanMessage):
            st.write(f"👤: {msg.content}")
        elif isinstance(msg, AIMessage):
            st.write(f"🤖: {msg.content}")