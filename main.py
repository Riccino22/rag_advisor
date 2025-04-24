import streamlit as st
import os
from dotenv import load_dotenv
from groq import Groq
import vectorstore as vs
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

load_dotenv()

chunks = vs.get_text_chunks()
vectorstore = vs.get_vectorstore(chunks)
conversation_chain = vs.get_conversation_chain(vectorstore)

st.title("Preguntame algo")
question = st.audio_input("Pregunta algo")

if not 'chat_history' in st.session_state:
    st.session_state.chat_history = []
if not 'conversation' in st.session_state:
    st.session_state.conversation = ""

if question:
    question_file = (question.name, question.getvalue())
    print(2+2)

    client = Groq()
    # filename = os.path.dirname(__file__) + "/audio.m4a"

    transcription = client.audio.transcriptions.create(
      file=question_file,
      model="whisper-large-v3",
      response_format="verbose_json"
    )

    

    prompt = f"""
    Responde la siguiente pregunta del usuario: {transcription.text}
    """

    if st.session_state.conversation.strip():
        conversation_prompt = f"""
      * Y toma en cuenta el contexto de la conversación:
      {st.session_state.conversation}
      """
        prompt += conversation_prompt

    response = conversation_chain({'question': prompt, 'chat_history': st.session_state['chat_history']})
    #messages = [
    #    HumanMessage(content=transcription.text),
    #    SystemMessage(content="Processing your question..."),
    #]  
    # response = conversation_chain()
    st.write(response['answer'])
    
    # Corregir cómo se guarda el historial
    if 'chat_history' in response:
        for message in response['chat_history']:
            st.session_state['chat_history'].append(message)
    else:
        # Fallback: guardar la pregunta y respuesta actual
        st.session_state['chat_history'].append(HumanMessage(content=transcription.text))
        st.session_state['chat_history'].append(AIMessage(content=response['answer']))
        
    print(transcription.text)
    
    # Mostrar historial para depuración
    st.session_state.conversation = ""
    st.subheader("Historial de Chat:")
    for msg in st.session_state['chat_history']:
        if isinstance(msg, HumanMessage):
            #st.write(f"👤: {msg.content}")
            pass
            #st.session_state.conversation += f"🤖 Bot: {msg.content}\n"
        elif isinstance(msg, AIMessage):
            st.write(f"🤖: {msg.content}")
            st.session_state.conversation += f"👤 Usuario: {msg.content}\n"