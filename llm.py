from sentence_transformers import SentenceTransformer, util
from langchain_groq import ChatGroq
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import HumanMessage, AIMessage
from langchain_experimental.tools.python.tool import PythonREPLTool
from langchain_experimental.agents.agent_toolkits.python.base import create_python_agent
from langchain.memory import ChatMessageHistory
import vectorstore as vs
import ast
import pandas as pd
from pathlib import Path
import json
import re

model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

def run_agent(agent, prompt):
        try:
            return agent.run(prompt)
        except ValueError as e:
             print(e)
             return run_agent(agent, prompt)

# Función para manejar la conversación
def chat(user_prompt, selected_model, chat_history):
    chat_model = ChatGroq(model_name=selected_model, temperature=0.1)
    agent_exec = create_python_agent(
    llm=chat_model,
    tool=PythonREPLTool(),
    verbose=True,
    handle_parsing_errors=True
    )

    # Inicializa la memoria
    memory = ConversationBufferMemory(chat_memory=ChatMessageHistory(messages=chat_history))


    # Crea una cadena de conversación con memoria
    conversation = ConversationChain(
        llm=chat_model, 
        memory=memory,
        verbose=True  # Esto muestra los detalles del proceso
    )
    
    manual_dataframe = vs.get_embeddings_dataframe("datasets/embeddings.csv")

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
                "resume": "Un resumen corto y conciso de lo que sabes del manual de unas 100 palabras aproximadamente"
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
            response_content = response.replace("```json", "").replace("```", "")
            response_content = re.search(r'\{.*?\}', response_content, re.DOTALL).group()
            model_response_json = json.loads(response_content)
            agent_response = run_agent(agent_exec, f"""
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

            # IMPORTANTE:
            - No menciones los archivos csv en tu respuesta final
            - No menciones al modelo anterior como un agente aparte, ya que tu respuesta es un complemento de la del anterior
            - En caso de que no sepas la respuesta, da una respuesta natural sin demasiados detalles.
            """)
            memory.chat_memory.messages[-1] = AIMessage(content=agent_response)
            return agent_response, memory.chat_memory.messages
        
        except (json.decoder.JSONDecodeError, AttributeError):
            return response, memory.chat_memory.messages
        
    except Exception as e:
        print(f"------------ Error ------------")
        print(e)
        return f"Error al generar la respuesta: '{e}', prueba cambiando el modelo", memory.chat_memory.messages