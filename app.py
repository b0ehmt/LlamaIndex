import os
import streamlit as st

from llama_index import GPTVectorStoreIndex, SimpleDirectoryReader, LLMPredictor, PromptHelper, ServiceContext
from langchain.llms.openai import OpenAI


# Page title
app_title = "LlamaIndex Knowledge Base"
st.set_page_config(page_title=app_title, page_icon="🔎")

# API-Key input
openai_api_key = st.sidebar.text_input(
    label="Geheimer OpenAI API-Key:",
    placeholder="API-Key einfügen, sk-...",
    type="password")

# Data directory
directory_path = "./data"

# Data sidebar display
files = os.listdir(directory_path)
st.sidebar.write(f"Geladene Dateien: {len(files)}")
st.sidebar.table(files)


def get_response(query, directory_path, openai_api_key):
    llm_predictor = LLMPredictor(
        llm=OpenAI(openai_api_key=openai_api_key, temperature=0, model_name="text-davinci-003"))

    # Prompt parameters and helper
    max_input_size = 4096
    num_output = 300
    max_chunk_overlap = 20
    chunk_overlap_ratio = 0.1

    prompt_helper = PromptHelper(max_input_size, num_output, chunk_overlap_ratio, max_chunk_overlap)

    if os.path.isdir(directory_path):
        # Loading in documents from the data directory
        documents = SimpleDirectoryReader(directory_path).load_data()
        service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, prompt_helper=prompt_helper)
        index = GPTVectorStoreIndex.from_documents(documents, service_context=service_context)

        query_engine = index.as_query_engine()
        response = query_engine.query(query)
        if response is None:
            st.error("Ups! Kein Ergebnis gefunden")
        else:
            st.success(response)
    else:
        st.error(f"Ungültiges Verzeichnis: {directory_path}")


# Prompt interface
st.title(app_title)
st.text("Durchsuche die geladenen Dateien mit Hilfe von LlamaIndex und InstructGPT.")

query = st.text_input("", "", placeholder="🔎 Was möchtest du wissen?")

# Adding further prompting instructions to query
query = f"Beantworte die folgende Frage in weniger als 3 Sätzen oder maximal 200 Zeichen. Verwende für die Beantwortung der" \
        f"Frage den gegebenen Kontext. Das ist die Frage: {query}"


# Submit button and main function execution
if st.button("Absenden", type="primary"):
    if not query.strip():
        st.error(f"Bitte geben Sie die Suchanfrage ein.")
    else:
        try:
            if len(openai_api_key) > 0:
                with st.spinner('Überlege...'):
                    get_response(query, directory_path, openai_api_key)
            else:
                st.error(f"Geben Sie einen gültigen API-Key ein")
        except Exception as e:
            st.error(f"Fehler: {e}")
