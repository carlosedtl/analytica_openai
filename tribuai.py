import streamlit as st
from openai import OpenAI
from pinecone.grpc import PineconeGRPC as Pinecone
import time
import pickle
import pathlib as Path
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader
import os

api_key_openai=os.getenv('OPENAI_API')
api_key_pinecone=os.getenv('PINECONE_API')

pc = Pinecone(api_key=api_key_pinecone)

index_name = "semantic-search-openai"
index = pc.Index(index_name)
MODEL = "text-embedding-3-small"

client = OpenAI(
    api_key=api_key_openai
)  # get API key from platform.openai.com

def response_generator(prompt):
    for word in prompt.split():
        yield word + " "
        time.sleep(0.05)

def query_pinecone(query):
    query = query  # Your query
    top_k = 3  # Number of relevant chunks to retrieve
    
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
            "Usa el siguiente texto para contestar la pregunta: "
            f"Pregunta: {query}\n\n"
            f"Text: {relevant_chunks[0]}\n\n"
            f"Text: {relevant_chunks[1]}\n\n"
            f"Text: {relevant_chunks[2]}\n\n"
            f"Text: {relevant_chunks[3]}\n\n"
            f"Text: {relevant_chunks[4]}\n\n"                
            f"Cuando sea posible contesta con al menos 2 parrafos, genera respuestas unicamente con base en los textos"
        )
    

    # Generate an augmented response using the LLM
    response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=2000,
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


# Streamlit app
def main():

    file_path='/Users/charlie/Library/Mobile Documents/com~apple~CloudDocs/Proyecto_Omega/code/Asistente/config.yaml'

    with open(file_path,'rb') as file:
        config = yaml.load(file, Loader=SafeLoader)

    # Pre-hashing all plain text passwords once
    # stauth.Hasher.hash_passwords(config['credentials'])

    authenticator = stauth.Authenticate(
        config['credentials'],
        config['cookie']['name'],
        config['cookie']['key'],
        config['cookie']['expiry_days']
    )

    try:
        authenticator.login()
    except Exception as e:
        st.error(e)

    if st.session_state['authentication_status']:
        authenticator.logout(location='sidebar')
        st.sidebar.text(f'Hola {st.session_state["name"]}')
        st.title("🤖 Genious Asistente con IA 🤖")
        
        st.sidebar.title("Navegación")
        menu = st.sidebar.radio(
                                "Ir a",
                                ("Información Tributaria")
                                )
                
        if menu == "Información Tributaria":                                     
            chat_bot_info_general()

    elif st.session_state['authentication_status'] is False:
        st.error('Username/password is incorrect')
    elif st.session_state['authentication_status'] is None:
        st.warning('Please enter your username and password')

     
# Run the app
if __name__ == "__main__":
    main()