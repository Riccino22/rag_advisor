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
import json
import time

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
    try:
        print("¡Bienvenido al chat! (Escribe 'salir' para terminar)")
    
        question = model.encode([user_prompt], show_progress_bar=True, batch_size=64)
        manual_dataframe['similarity'] = manual_dataframe['embedding'].apply(lambda x: util.cos_sim(question, ast.literal_eval(x)))
        manual_dataframe.sort_values('similarity', ascending=False, inplace=True)
        manual_result = manual_dataframe.head(4)['text'].to_list()
        
        # Obtener respuesta del modelo
        final_prompt = f"""
            # CONTEXTO
            Eres un experto en asesoramiento de datos empresariales.

            # INFORMACIÓN DE REFERENCIA
            Esta información ha sido extraída del manual de la empresa:
            {manual_result}

            # INSTRUCCIONES
            1. Si la pregunta del usuario se relaciona con la empresa, responde basándote en la información proporcionada.
            2. Si es un saludo o mensaje ambiguo, responde naturalmente sin forzar la información empresarial.
            3. Si no conoces la respuesta basada en el manual, proporciona un JSON con esta estructura exacta:
            ```json
            {{
                "resume": "Un resumen corto y conciso de lo que sabes del manual"
            }}

            # IMPORTANTE, MUY IMPORTANTE
            4. El json con el resumen es solamente si no conoces la respuesta a la pregunta del usuario. Si la conoces, no debes incluir el resumen en json
            5. En caso de que no sepas la respuesta y debas responder con el json, no incluyas ningun texto fuera de las llaves del json (Ejemplo: No escribas fuera del json cosas como "Desafortunadamente,..." o "No tengo información, ..."). Todo eso lo puedes incluir dentro del json pero no por fuera.


            * La pregunta del usuario es: '{user_prompt}'
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
        try:
            model_response_json = json.loads(response.replace("```json", "").replace("```", ""))
            agent_response = run_agent(f"""
            # CONTEXTO
            El usuario preguntó: '{user_prompt}'
            El modelo respondió con este resumen: '{model_response_json['resume']}'

            # INSTRUCCIONES
            1. Si la respuesta del modelo fue correcta y completa, devuélvela exactamente igual.
            2. Si la respuesta fue insuficiente y la pregunta es relevante para el contexto empresarial:
            - Utiliza PythonREPL y pandas para analizar los archivos en 'datasets/'
            - Archivos disponibles: {list(folder.glob("*.csv"))}

            # INFORMACIÓN DE LOS DATASETS
            {info_dataframes}'

            # ACCIÓN REQUERIDA
            - Si la pregunta no se relaciona con los archivos o la empresa, simplemente responde al mensaje del usuario.
            - Mantén la respuesta concisa para optimizar velocidad.
            """)
            memory.chat_memory.messages[-1] = AIMessage(content=agent_response)
            return agent_response, memory.chat_memory.messages
        
        except json.decoder.JSONDecodeError:
            return response, memory.chat_memory.messages
    
    except Exception as e:
        print(f"------------ Error ------------")
        print(e)
        time.sleep(2)
        memory.clear()
        return chat(user_prompt)




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