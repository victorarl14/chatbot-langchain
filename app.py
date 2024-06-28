import streamlit as st
from PIL import Image
from langchain import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pathlib
import textwrap
import google.generativeai as genai
from IPython.display import display
from IPython.display import Markdown
from langchain_community.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from datetime import datetime
import requests
from streamlit_lottie import st_lottie
import os
import time
import fitz  # PyMuPDF
from pinecone import Pinecone, ServerlessSpec
from langchain_community.vectorstores import Pinecone as LC_Pinecone
from langchain_google_genai import ChatGoogleGenerativeAI
from sentence_transformers import SentenceTransformer
from langchain.docstore.document import Document  # Asegúrate de importar Document

# API KEY DE Gemini
GOOGLE_API_KEY = 'AIzaSyA-TPevJZj0J7hgWyipc8WCaju06j6w5og'
PINECONE_API_KEY = '8ab68be5-cb75-497b-9c58-5f09c16465aa'
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-pro')

# Configuración de la clave API de Google como variable de entorno
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

# URL del Lottie
url = 'https://lottie.host/ab476d01-901e-442a-9eca-1eec130bac01/nDPlPSxi1W.json'

# Carga el lottie de la url
def load_lottie(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie = load_lottie(url)

st.set_page_config(page_title="ChatBot", page_icon="🤖", layout="wide")

# ESTRUCTURA BÁSICA DEL CONTENEDOR DEL CHAT
html_content = """
<div id='chat-container' style='border: 2px solid #000; padding: 10px; border-radius: 5px; background-color: #f9f9f9;'>
    <p style='color: black;'>Bienvenido al ChatBot ¿Cómo puedo ayudarte hoy?</p>
</div>
"""

# Titulo de la App
with st.container():
    first_column, second_column, third_column = st.columns(3)
    with first_column:
        st.empty()
    with second_column:
        st.header("🤖 ChatBot Xyra 🤖")
    with third_column:
        st.empty()

# BOT Imagen
with st.container():
    first_column, second_column, third_column = st.columns(3)
    with first_column:
        st.empty()
    with second_column:
        st_lottie(lottie, height=200)
    with third_column:
        st.write("")

# Contenedor del chat
with st.container():
    st.write("---")
    first_column, second_column, third_column = st.columns((1, 3, 1))
    with first_column:
        st.empty()
    with second_column:
        chat_container = st.empty()  # Placeholder for the chat container
        chat_container.markdown(html_content, unsafe_allow_html=True)
    with third_column:
        st.empty()

# Saltos de línea
st.write("")
st.write("")

def save_conversation_to_txt(conversation_history):
    try:
        # Define the file path and name
        file_path = "conversation_history.txt"
        
        # Write the conversation history to the file
        with open(file_path, "w") as file:
            for message in conversation_history:
                # Remove HTML tags for a cleaner text file
                clean_message = message.replace("<p>", "").replace("</p>", "").replace("<b>", "").replace("</b>", "")
                file.write(clean_message + "\n\n")  # Añade doble salto de línea para separación
        
        # Provide a link to download the file
        with open(file_path, "rb") as file:
            btn = st.download_button(
                label="Descargar conversación",
                data=file,
                file_name="conversation_history.txt",
                mime="text/plain"
            )
    except Exception as e:
        st.error(f"Error al guardar la conversación: {e}")

# Footer del Chat
with st.container():
    first_column, second_column, third_column = st.columns((1, 3, 1))
    with first_column:
        st.empty()
    with second_column:
        pdf_obj = st.file_uploader("Carga tu documento", type="pdf")
        
        with st.form(key='my_form', clear_on_submit=True):
            user_question = st.text_input("Escribe tu mensaje aquí:", key="user_question")
            submit_button = st.form_submit_button(label='Enviar')
        
        if st.button("Guardar Conversación"):
            if 'conversation_history' in st.session_state and st.session_state.conversation_history:
                save_conversation_to_txt(st.session_state.conversation_history)
    with third_column:
        st.empty()

# List to hold the conversation history
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []

@st.cache_resource
def create_embeddings(pdf, pinecone_api_key):
    try:
        doc = fitz.open(stream=pdf.read(), filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text)

        documents = [Document(page_content=chunk) for chunk in chunks]

        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

        os.environ["PINECONE_API_KEY"] = pinecone_api_key
        pinecone_api_key = os.environ.get("PINECONE_API_KEY")

        if pinecone_api_key:
            try:
                pc = Pinecone(api_key=pinecone_api_key)

                index_name = "langchain-project"

                existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]
                if index_name not in existing_indexes:
                    pc.create_index(
                        name=index_name,
                        dimension=384,
                        metric="cosine",
                        spec=ServerlessSpec(cloud="aws", region="us-east-1")
                    )

                    while not pc.describe_index(index_name).status["ready"]:
                        time.sleep(1)

                vector_store = LC_Pinecone.from_documents(
                    documents,
                    embeddings,
                    index_name=index_name
                )

                return vector_store

            except Exception as e:
                st.error(f"Error al procesar el archivo PDF: {e}")

    except Exception as e:
        st.error(f"Error al procesar el archivo PDF: {e}")

def to_markdown(text):
    text = text.replace('•', '  *')
    return textwrap.indent(text, '> ', predicate=lambda _: True)

if pdf_obj:
    if GOOGLE_API_KEY and PINECONE_API_KEY:
        knowledge_base = create_embeddings(pdf_obj, PINECONE_API_KEY)

        if submit_button and user_question:  # Check if the submit button was clicked and there's a question
            try:
                model = ChatGoogleGenerativeAI(google_api_key=GOOGLE_API_KEY, model="gemini-pro")

                prompt_template = """
                    Answer the question as detailed as possible from the provided context,
                    make sure to provide all the details, answer me in Spanish,
                    try to give me an answer that's not just "there's no context",
                    Give long and detailed answers, in several paragraphs if necessary. 
                    If there is something that is not in the document, try to complement it with your knowledge.
                    don't provide the wrong answer\n\n
                    Context:\n {context}?\n
                    Question: \n{question}\n
                    Answer:
                """
                prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

                chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

                docs = knowledge_base.similarity_search(user_question, 10)
                respuesta = chain.run(input_documents=docs, question=user_question)

                # Add question and answer to the conversation history
                st.session_state.conversation_history.append(f"<p><b>Pregunta:</b> {user_question}</p>")
                st.session_state.conversation_history.append(f"<p><b>Respuesta:</b> {respuesta}</p>")

                # Update the chat container with the entire conversation
                updated_html_content = """
                <div id='chat-container' style='border: 2px solid #000; padding: 10px; border-radius: 5px; background-color: #FFFFFF; color: black'>
                """
                for message in st.session_state.conversation_history:
                    updated_html_content += message
                updated_html_content += "</div>"

                chat_container.markdown(updated_html_content, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Error al ejecutar la consulta: {e}")

    else:
        st.warning("Por favor, configura las API keys en tu entorno.")