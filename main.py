import streamlit as st
import os
from dotenv import load_dotenv
from groq import Groq
import vectorstore as vs
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from llm import chat

load_dotenv()

# chunks = vs.get_text_chunks()
# vectorstore = vs.get_vectorstore(chunks)
# conversation_chain = vs.get_conversation_chain(vectorstore)

st.title("InnovaTech Solutions Chat")
question_audio = st.audio_input("Pregunta algo")
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
        response, messages = chat(user_input)
    
    st.subheader("Historial de Chat:")
    for msg in messages[::-1]:
        if isinstance(msg, HumanMessage):
            st.write(f"👤: {msg.content}")
        elif isinstance(msg, AIMessage):
            st.write(f"🤖: {msg.content}")