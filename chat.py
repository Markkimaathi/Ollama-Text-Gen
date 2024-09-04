import ollama
import streamlit as st
import pickle
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader

st.set_page_config(
    page_title="Chat playground",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("NexusBot: News Research Tool 📈")
st.sidebar.title("News Article URLs")


urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    if url.strip():  
        urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")
file_path = "faiss_store_ollama.pkl"

main_placeholder = st.empty()

models_info = ollama.list()
available_models = [model["name"] for model in models_info.get("models", [])]

if available_models:
    selected_model = st.selectbox(
        "Pick a model available locally on your system ↓", available_models
    )
else:
    st.warning("You have not pulled any model from Ollama yet!", icon="⚠️")
    if st.button("Go to settings to download a model"):
        st.page_switch("pages/03_⚙️_Settings.py")

if process_url_clicked and available_models:
    if urls:  
        try:
            loader = UnstructuredURLLoader(urls=urls)
            main_placeholder.text("Data Loading... Started... ✅✅✅")
            data = loader.load()

            text_splitter = RecursiveCharacterTextSplitter(
                separators=['\n\n', '\n', '.', ','],
                chunk_size=1000
            )
            main_placeholder.text("Text Splitting... Started... ✅✅✅")
            docs = text_splitter.split_documents(data)

           
            with open(file_path, "wb") as f:
                pickle.dump(docs, f)

            main_placeholder.success("Processing complete. You can now ask questions.")
        except Exception as e:
            st.error(f"An error occurred during processing: {e}")
    else:
        st.warning("Please enter at least one valid URL.")


query = main_placeholder.text_input("Question: ")

if query and available_models:
    if os.path.exists(file_path):
        try:
            with open(file_path, "rb") as f:
                docs = pickle.load(f)
            docs_to_send = docs[:3]  
            response = ollama.chat(
                model=selected_model,
                messages=[
                    {"role": "user", "content": f"Based on these documents, answer this question: {query}. Documents: {docs_to_send}"}
                ]
            )

      
            st.write("Response received from the model:")
            st.json(response) 

            
            if 'completion' in response:
                st.header("Answer")
                st.write(response['completion'])
            else:
                st.error("The response does not contain a 'completion' field. Please check the response structure.")
        except Exception as e:
            st.error(f"An error occurred while retrieving the answer: {e}")
    else:
        st.warning("The documents do not exist. Please process the URLs first.")
elif not available_models:
    st.warning("No models are available to process your query.")
