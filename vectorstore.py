from langchain.text_splitter import CharacterTextSplitter
from sentence_transformers import SentenceTransformer
import pandas as pd


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=300,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def convert_text_to_embeddings(text):
    model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    chunks = get_text_chunks(text)
    df = pd.DataFrame(chunks, columns=['text'])
    embeddings = model.encode(chunks, show_progress_bar=True, batch_size=64)
    df['embedding'] = embeddings.tolist()
    df.to_csv('manual/embeddings.csv', index=False)

def get_embeddings_dataframe(path):
    
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:

        with open("manual/manual.txt", "r") as file:
            text = file.read()
        convert_text_to_embeddings(text)
        return get_embeddings_dataframe(path)
    
    return df