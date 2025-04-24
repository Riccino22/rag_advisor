from sentence_transformers import SentenceTransformer, util
from langchain_groq import ChatGroq
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import HumanMessage, AIMessage
from langchain_experimental.tools.python.tool import PythonREPLTool
from langchain_experimental.agents.agent_toolkits.python.base import create_python_agent
import ast
import pandas as pd
from pathlib import Path

model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
chat_model = ChatGroq(model_name="meta-llama/llama-4-scout-17b-16e-instruct", temperature=0.1)
manual_dataframe = pd.read_csv("datasets/embeddings.csv")
agent_exec = create_python_agent(
    llm=chat_model,
    tool=PythonREPLTool(),
    verbose=True,
    handle_parsing_errors=True
)

# Inicializa la memoria
memory = ConversationBufferMemory()


# Crea una cadena de conversación con memoria
conversation = ConversationChain(
    llm=chat_model, 
    memory=memory,
    verbose=True  # Esto muestra los detalles del proceso
)

def run_agent(prompt):
        try:
            return agent_exec.run(prompt)
        except ValueError as e:
             print(e)
             return run_agent(prompt)

# Función para manejar la conversación
def chat(user_prompt):
    print("¡Bienvenido al chat! (Escribe 'salir' para terminar)")
    
    while True:
        # user_prompt = input("Tú: ")
        question = model.encode([user_prompt], show_progress_bar=True, batch_size=64)
        manual_dataframe['similarity'] = manual_dataframe['embedding'].apply(lambda x: util.cos_sim(question, ast.literal_eval(x)))
        manual_dataframe.sort_values('similarity', ascending=False, inplace=True)
        manual_result = manual_dataframe.head(4)['text'].to_list()
        
        # Obtener respuesta del modelo
        final_prompt = f"""
        Eres un experto en asesoramiento de los datos de la empresa.
        Y esta información extraida del manual de la empresa:
        {manual_result}

        * Voy a enseñarte una pregunta del usuario para que la respondas basado en la información que acabo de darte. Si la pregunta o el mensaje del usuario no tiene que ver con la empresa (como un saludo o un mensaje ambiguo), responde con normalidad sin basarte en esa información (a menos que sea necesario)

        Esta es la pregunta del usuario que debes responder: '{user_prompt}'
        """
        response = conversation.predict(input=final_prompt)
        memory.chat_memory.messages[-2] = HumanMessage(content=user_prompt)
        folder = Path("datasets")
        info_dataframes = ""

        for f in list(folder.glob("*.csv")):
            df = pd.read_csv(f"datasets/{f.name}")
            info_dataframes += f"""
                * Nombre del archivo: {f.name}
                * Columnas: {df.columns}
                * Filas x Columnas: {df.shape}
                * Algunos registros extraidos: '{df.sample(4).to_dict(orient='records')}' 
            """
        agent_response = run_agent(f"""
                  
            El usuario hizo la siguiente pregunta: '{user_prompt}'
            Y esta fue la respuesta del modelo: 
            '{response}'

            Primero, analiza si la respuesta del modelo y si la respuesta fue correcta para la pregunta del usuario entonces envia como respuesta exactamente la respuesta del modelo que te acabo de mostrar.

            En caso de que la respuesta del modelo sea de desconocimiento o no haya respondido satisfactoriamente a la pregunta del usuario ya que no se encuentra respuesta en el manual de la empresa, y sea una pregunta válida para el contexto empresarial, entonces usa PythonREPL y pandas para leer las columnas o las filas de los archivos csv de la carpeta 'datasets' (los archivos disponibles de la carpeta 'datasets' son: {list(folder.glob("*.csv"))}) para averiguar la pregunta del usuario.

            Esta es información base de los archivos csv en caso de que sea de utilidad:
            {info_dataframes}

            Si la pregunta del usuario no tiene nada que ver con esos archivos o con la empresa en general, genera una respuesta basada en su mensaje simplemente.
        """)
        memory.chat_memory.messages[-1] = AIMessage(content=agent_response)
        return agent_response, memory.chat_memory.messages

# Ejecutar el chat
if __name__ == "__main__":
    chat()
# question = model.encode(['Hola'], show_progress_bar=True, batch_size=64)
# response = chat_model.invoke({'input': 'Hola', 'chat_history': []})


"""


    # if st.session_state.conversation.strip():
    #    conversation_prompt = f"
    #  * Y toma en cuenta el contexto de la conversación:
    #  {st.session_state.conversation}
    #  ""
    #    prompt += conversation_prompt

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
    
"""