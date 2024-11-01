# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 16:42:11 2024

@author: torrec10
"""

import streamlit as st
from openai import OpenAI
from pinecone.grpc import PineconeGRPC as Pinecone
import time
import mysql.connector
from mysql.connector import Error
from io import BytesIO
import matplotlib.pyplot as plt


api_key="OPENAI_API"
pc = Pinecone(api_key="PINECONE_API")

index_name = "semantic-search-openai"
index = pc.Index(index_name)
MODEL = "text-embedding-3-small"

client = OpenAI(
    api_key=api_key
)  # get API key from platform.openai.com

# Función para guardar gráficos en session_state
def save_chart():
    buf = BytesIO()
    plt.savefig(buf, format="png")
    st.session_state["charts"].append(buf)

def create_connection():
    try:
        connection = mysql.connector.connect(
            host='localhost',
            user='root',
            password='carlos1245',
            database='ventas'
        )
        if connection.is_connected():
            return connection
    except Error as e:
        print(f"Error al conectar: {e}")
        return None

# Función para ejecutar la consulta SQL
def execute_query(connection, query):
    try:
        cursor = connection.cursor()
        cursor.execute(query)
        return cursor.fetchall()
    except Error as e:
        print(f"Error ejecutando la consulta: {e}")
        return None

def response_generator(prompt):
    for word in prompt.split():
        yield word + " "
        time.sleep(0.05)

def query_pinecone(query):
    query = query  # Your query
    top_k = 5  # Number of relevant chunks to retrieve
    
    # Perform a similarity search using the query
    query_embedding = client.embeddings.create(input=query, model=MODEL).data[0].embedding
    res = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True,
    )
    
    # Extract the relevant chunks from the search results
    relevant_chunks = [result['metadata']['text'] for result in res['matches']]
    
    # Combine the relevant chunks into a single prompt for the LLM
    prompt = (
        "Use the following text to answer the question: "
        f"Question: {query}\n\n"
        f"Text: {relevant_chunks[0]}\n\n"
        f"Text: {relevant_chunks[1]}\n\n"
        "Answer:"
    )
    
    # Generate an augmented response using the LLM
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
        max_tokens=1000,
    )
    return(response.choices[0].message.content)


def chat_bot_info_general():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Accept user input
    if prompt := st.chat_input("Cómo te puedo ayudar?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
    
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            query = query_pinecone(prompt)
            response= st.write_stream(response_generator(query))
            
         # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})   



def chat_bot_sales():
    if "messages" not in st.session_state:
        st.session_state.messages = []
        
    if "charts" not in st.session_state:
        st.session_state.charts = []
        
    # Display chat messages from history on app rerun
    for chart in st.session_state.charts:
       st.image(chart, use_column_width=True)
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    
    # Accept user input
    if user_input := st.chat_input("Cómo te puedo ayudar?"):
        
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        
        # Display user message in chat message container
        with st.chat_message("user"):                                    
            st.markdown(user_input)                   
                                                        
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            connection = create_connection()
            
            if connection:

                prompt =  """
                    La base de datos tiene una tabla llamada 'ventas' con las columnas: 
                    fecha date 
                    producto varchar(100) 
                    cantidad int 
                    precio_unitario decimal(10,2) 
                    pais varchar(100) 
                    canal varchar(45) 
                    monto_vendido decimal(10,2) La moneda del monto vendido es colones
                    Convierte la siguiente pregunta en una consulta SQL válida, no agregues nada más que el script de consulta                
                    Los productos pueden estar en singular o plural                                
                    Pregunta:
                 """
                prompt = prompt + " " + user_input

                client = OpenAI(api_key=api_key)
                response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                      {"role": "system", "content": "You are an assistant that converts natural languge to SQL queries"},                                      
                      {"role": "user", "content": [{"type": "text", "text": prompt},]}
                         ] ,
                      max_tokens=2000,
                )                                
                sql =   response.choices[0].message.content             
                sql= sql.replace("```sql", "")
                sql= sql.replace("```", "") 
                sql= sql.replace(";", "")                        
                results = execute_query(connection, sql)                        
                
                
                prompt2 = "The following list contains data to be plotted, generate the streamlit code to plote this data, only generate code" + str(results) 
                client = OpenAI(api_key=api_key)
                response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                      {"role": "system", "content": """You are a programmer that codes for python and streamlit, 
                                                        you will recieve data in form of a list,                                                    
                                                        Always add values on top of the bars,
                                                        If there is no valid data put: "No se encontraton datos",
                                                        Resize the the font according to the size of the chart,
                                                        Generate a chat title accoring to the data requested,
                                                        only generate code"""}, 
                      {"role": "user", "content": [{"type": "text", "text": prompt2},]}
                         ] ,
                      max_tokens=800,
                )                                
                                                                                                                                              
                sentencia =response.choices[0].message.content
                cleaned_answer = sentencia.replace("```python", "").replace("```", "")  # Remove code block markers                        
                exec_globals = {"st": st, "plt": plt}
                exec(cleaned_answer, exec_globals)
              
              # Guardar el gráfico generado en session_state
                save_chart()
                connection.close()
                
        #st.session_state.messages.append({"role": "assistant", "content": exec(cleaned_answer)})   
        
        with st.chat_message("assistant"):        
        
                prompt3 = """The following list contains data to be plotted,
                            always answer in spanish,
                            generate one pharagraph with insigts of the data""" + str(results) 
                client = OpenAI(api_key=api_key)
                response_comentario = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                      {"role": "system", "content": """You are a data analyst that provides description of the data, provide an small pharagraph"""}, 
                      {"role": "user", "content": [{"type": "text", "text": prompt3},]}
                      
                         ] ,
                      temperature=0.0,
                      max_tokens=800,
                )
                st.write(response_comentario.choices[0].message.content)
                st.session_state.messages.append({"role": "assistant", "content": response_comentario.choices[0].message.content})    

# Streamlit app
def main():
       
    st.title("🤖 Genious Asistente con IA 🤖")
    
    st.sidebar.title("Navegación")
    menu = st.sidebar.radio(
        "Ir a",
        ("Ventas", "Información General")
    )
    
    # Render the selected page based on the menu selection

#############################################################   SALES    ##################################    
    if menu == "Ventas":                                     
       chat_bot_sales()
    
#############################################################   GENERAL    ##################################    
    elif menu == "Información General":
          
        chat_bot_info_general()
    
    
# Run the app
if __name__ == "__main__":
    main()
