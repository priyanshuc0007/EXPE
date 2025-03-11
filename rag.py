import sys
import pysqlite3
import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from dotenv import load_dotenv

sys.modules['sqlite3'] = pysqlite3

# Load environment variables
load_dotenv()

# Retrieve Google API Key from environment variables
google_api_key = os.getenv("GOOGLE_API_KEY")
if not google_api_key:
    st.error("Google API Key not found. Please set it in the .env file.")
    st.stop()

# Initialize embedding model and vector store
embedding_model = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
vector_store = Chroma("current_affairs", embedding_model, persist_directory="./chroma_db")

# Define custom prompt template
template = """
Use the following context to answer the user's question.
If the answer isn't present in the context, respond with "I'm not sure about that."

Context:
{context}

Question:
{question}

Answer in a simple and easy-to-understand way:
"""

custom_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=template
)

# Initialize language model and retrieval chain
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.1)
retrieval_chain = ConversationalRetrievalChain.from_llm(
    llm, retriever=vector_store.as_retriever(topk=3), return_source_documents=True,
    combine_docs_chain_kwargs={"prompt": custom_prompt}
)

# Initialize conversation history
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# Function to process PDF upload
def process_pdf_upload(uploaded_file):
    if uploaded_file is None:
        st.warning("No file uploaded.")
        return

    pdf_text = extract_text_from_pdf(uploaded_file)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=500)
    text_chunks = text_splitter.split_text(pdf_text)

    vector_store.add_texts(text_chunks)
    vector_store.persist()

    st.success("PDF processed and added to the vector store.")

# Function to get response from the chatbot
def get_response(user_query):
    response = retrieval_chain({"question": user_query, "chat_history": st.session_state.conversation_history})
    st.session_state.conversation_history.append((user_query, response["answer"]))
    return response["answer"]

# Streamlit UI
st.title("PDF Chatbot with Streamlit")

# Sidebar for PDF upload
st.sidebar.header("Upload PDF")
uploaded_file = st.sidebar.file_uploader("Choose a PDF file", type="pdf")
if uploaded_file is not None:
    process_pdf_upload(uploaded_file)

# Main chat interface
st.header("Chat with your PDF")
user_input = st.text_input("Enter your question:")
if st.button("Send"):
    if user_input:
        response = get_response(user_input)
        st.text_area("Chatbot Response:", value=response, height=200)
    else:
        st.warning("Please enter a question.")

# Display conversation history
if st.session_state.conversation_history:
    st.subheader("Conversation History")
    for i, (question, answer) in enumerate(st.session_state.conversation_history, 1):
        st.markdown(f"**Q{i}:** {question}")
        st.markdown(f"**A{i}:** {answer}")
