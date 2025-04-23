import streamlit as st
import os
from dotenv import load_dotenv
from groq import Groq
import vectorstore as vs

load_dotenv()

st.title("Preguntame algo")
question = st.audio_input("Pregunta algo")

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

    print(transcription.text)