
import streamlit as st
from openai import OpenAI
from pinecone.grpc import PineconeGRPC as Pinecone
import time


api_key="OPENAI_API"
pc = Pinecone(api_key="PINECONE_API")

index_name = "semantic-search-openai"
index = pc.Index(index_name)
MODEL = "text-embedding-3-small"

client = OpenAI(
    api_key=api_key
)  # get API key from platform.openai.com



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




# Streamlit app
def main():
       
    st.title("🤖 Genious Asistente con IA 🤖")
    
    st.sidebar.title("Navegación")
    menu = st.sidebar.radio(
        "Ir a",
        ("Información Tributaria")
    )
    
    # Render the selected page based on the menu selection

#############################################################   SALES    ##################################    
    if menu == "Información Tributaria":                                     
       chat_bot_info_general()
    
#############################################################   GENERAL    ##################################    
        
    
# Run the app
if __name__ == "__main__":
    main()