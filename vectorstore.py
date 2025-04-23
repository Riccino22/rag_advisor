from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from dotenv import load_dotenv
import os
from langchain_community.vectorstores import Chroma

load_dotenv()

# Generate chunks
def get_text_chunks():

    with open('manual.txt', 'r') as file:
        text = file.read()

    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=300,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


# create the vectorstore

def get_vectorstore():

    persist_directory = "./chroma_db"
    embeddings_model = SentenceTransformerEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings_model)
    return vectorstore

def get_conversation_chain(vectorstore):
    chat_model = ChatGroq(model_name="meta-llama/llama-4-maverick-17b-128e-instruct")
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=chat_model,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain