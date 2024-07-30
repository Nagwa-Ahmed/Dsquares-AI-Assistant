import streamlit as st
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import FlashrankRerank
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import joblib
import os
import nest_asyncio
import subprocess
import time
from langchain.text_splitter import MarkdownHeaderTextSplitter
import chromadb
from langchain_community.llms import Ollama

nest_asyncio.apply()

# Function to load the custom CSS
def load_css():
    with open("custom.css") as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Call the function to load custom CSS
load_css()

# Load configuration
with open('./config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
    config['pre-authorized']
)

authenticator.login()

if st.session_state["authentication_status"]:
    # Setting environment
    os.environ["OCR_ALL_PAGES"] = "False"
    os.environ["EXTRACT_IMAGES"] = "False"
    # Initialize Ollama client for local Llama3 model
    ollama_client = Ollama(model="llama3")
    embed_model = FastEmbedEmbeddings()

    def main():
        html_string = """
        <div style='background-color: #FFFFFF; padding: 50px; text-align: center; border-radius: 10px;'>
            <h1 style='color: #062F87; font-size: 65px; margin: 0;'>Welcome to Dsquares AI Assistant</h1>
        </div>
        """
        st.markdown(html_string, unsafe_allow_html=True)

        # Function to list files in the directory
        def list_files_in_directory():
            try:
                # List all files in the given directory
                files = [f for f in os.listdir('/home/project/project markdown') if os.path.isfile(os.path.join('/home/project/project markdown', f))]
                return files
            except Exception as e:
                print(f"An error occurred: {e}")
                return []

        file_list = list_files_in_directory()
        filter_dict = {"source": {"$in": file_list}}

        persist_directory = "/home/project/chromadb5"

        # Initialize Chroma vector store with existing collection (collection_name vimp)
        vectorstore = Chroma(collection_name='rag5', persist_directory=persist_directory, embedding_function=embed_model)

        retriever = vectorstore.as_retriever(search_kwargs={'k': 10, 'filter': filter_dict})

        compressor = FlashrankRerank()
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor, base_retriever=retriever
        )

        # Set up retriever from vector store
        custom_prompt_template = """Use the following pieces of information to answer the user's question.
        1- If you don't know the answer, just say that you don't know, don't try to make up an answer.
        2- Please provide a detailed answer to the following question without summarizing any parts. Ensure all relevant information is included: [Your question here]"
	3- When someone greets you , greet him and say that you are Dsquares AI Assistant and  you are here to help him to chat with Dsquares projects documents 
	4- When someone thanks you, say you're welcome! If you have any more questions or need further assistance, feel free to ask!

        Context: {context}
        Question: {question}

        Only return the helpful answer below and nothing else.
        """

        def set_custom_prompt():
            """
            Prompt template for QA retrieval for each vectorstore
            """
            prompt = PromptTemplate(template=custom_prompt_template, input_variables=['context', 'question'])
            return prompt

        prompt = set_custom_prompt()

        qa = RetrievalQA.from_chain_type(llm=ollama_client,
                                         chain_type="stuff",
                                         retriever=compression_retriever,
                                         return_source_documents=True,
                                         chain_type_kwargs={"prompt": prompt})

        # Implementing the Streamlit app
        if "messages" not in st.session_state:
            st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

        # Add New Chat and Logout buttons to the sidebar
        with st.sidebar:
            st.image("/home/project/image/dsquares logo.png", use_column_width=True)
            st.markdown('<div style="width:200px; height:350px;"></div>', unsafe_allow_html=True)
            if st.button("New Chat"):
                st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]
            if st.button("Logout"):
                authenticator.logout()
                st.experimental_rerun()

        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.chat_message(msg["role"], avatar="🧑‍💻").write(msg["content"])
            else:
                st.chat_message(msg["role"], avatar="🤖").write(msg["content"])

        if prompt := st.chat_input():
            st.chat_message("user", avatar="🧑‍💻").write(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})
            # Prepare the chat history to include in the prompt
            chat_history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.messages])

            # Concatenate chat history with the new prompt
            full_prompt = f"{chat_history}"

            # Show loading spinner while generating response
            with st.spinner('Generating response...'):
                # Call the QA system with the full prompt
                response = qa.invoke({"query": full_prompt})

            # Display the avatar icon and initialize response container
            st.chat_message("", avatar="🤖").write("")
            response_words = response['result'].split(' ')
            response_container = st.empty()

            # Stream the response word-by-word with a delay
            response_text = ""
            for word in response_words:
                if '\n' in word:
                    sub_words = word.split('\n')
                    for sub_word in sub_words[:-1]:
                        response_text += sub_word + " "
                        response_container.markdown(response_text)
                        time.sleep(0.01)
                        response_text += "\n"
                        response_container.markdown(response_text)
                        time.sleep(0.01)
                    response_text += sub_words[-1] + " "
                else:
                    response_text += word + " "
                    response_container.markdown(response_text)
                    time.sleep(0.01)

            st.session_state.messages.append({"role": "assistant", "content": response['result']})
        else:
            pass

    if __name__ == "__main__":
        main()

elif st.session_state["authentication_status"] is False:
    st.error('Username/password is incorrect')
elif st.session_state["authentication_status"] is None:
    st.warning('Please enter your username and password')
