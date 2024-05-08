import os
from dotenv import load_dotenv
import streamlit as st
import time
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import AzureChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import AzureAIDocumentIntelligenceLoader


# Load environment variables from .env file
load_dotenv()
OPENAI_ENAPOINT = os.getenv("AZURE_OAI_ENDPOINT")
OPENAI_API_KEY = os.getenv("AZURE_OAI_KEY")
OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OAI_DEPLOYMENT")
OPENAI_DEPLOYMENT_ENDPOINT = os.getenv("OAzure_AI_DEPLOYMENT_ENDPOINT")
OPENAI_DEPLOYMENT_VERSION = os.getenv("Azure_OAI_DEPLOYMENT_VERSION")
OPENAI_MODEL_NAME = os.getenv("Azure_OAI_MODEL_NAME")

OPENAI_ADA_EMBEDDING_DEPLOYMENT_NAME = os.getenv("Azure_OAI_EMBEDDING_DEPLOYMENT_NAME")
OPENAI_ADA_EMBEDDING_MODEL_NAME = os.getenv("Azurre_OAI_EMBEDDING_MODEL_NAME")
AZURE_DOC_INT_ENDPOINT=os.getenv("AZURE_DOC_INT_ENDPOINT")
AZURE_DOC_INT_Key=os.getenv("AZURE_DOC_INT_Key")


completionResponse=[]


# Streamlit App
def main():
    print(f'Start - Main - {time.ctime(time.time())}')
    st.set_page_config(
        page_title="CVS - Consumer Value Stores - Data Extraction",
        page_icon=":globe_with_meridians:",
        layout="wide",
    )
    print(f'Start - Title - {time.ctime(time.time())}')

    label = r'''$\textsf{\LARGE CVS - Consumer Value Stores Text Extraction}$'''
    st.write(label)
    # st.title("CDTS - Clinial Document Translation & Summarization")
    st.header('', divider='rainbow')

    # To keep Language selection in the same row
    col1, col2 = st.columns([8, 2])


    # File upload section
    with col1:
        uploaded_file = st.file_uploader(
            "Choose a good quality Scanned Document in correct orientation (upright) with less than 5 MB size.",
            accept_multiple_files=False,
            type=["jpg", "jpeg", "png","pdf"],
            key="file_uploader"
        )
   
    # Button to trigger processing
    with col2:
        st.markdown('<p style="color: red;">Hit submit after upload </p>', unsafe_allow_html=True)
    if col2.button("Submit", key="submit_button", use_container_width=True) and uploaded_file is not None:        
        if 'qaValue' in st.session_state:     
            del st.session_state['qaValue']      
        filePath =os.getenv("FILE_PATH")+ uploaded_file.name
        print("Path is:" +str(filePath))
        #init Azure OpenAI
        embeddings=AzureOpenAIEmbeddings(deployment=OPENAI_ADA_EMBEDDING_DEPLOYMENT_NAME,
                                model=OPENAI_ADA_EMBEDDING_MODEL_NAME,
                                azure_endpoint=OPENAI_DEPLOYMENT_ENDPOINT,
                                openai_api_type="azure",
                                chunk_size=1)
        
        loader = AzureAIDocumentIntelligenceLoader(
            api_endpoint=AZURE_DOC_INT_ENDPOINT, api_key=AZURE_DOC_INT_Key, file_path=filePath, api_model="prebuilt-layout"
        )
        documents = loader.load()
      
        db = FAISS.from_documents(documents=documents, embedding=embeddings)
        #save the embeddings into FAISS vector store
        db.save_local("./dbs/documentation/faiss_index")
       
        llm = AzureChatOpenAI(deployment_name=OPENAI_DEPLOYMENT_NAME,
                  model_name=OPENAI_MODEL_NAME,
                  azure_endpoint=OPENAI_DEPLOYMENT_ENDPOINT,
                  openai_api_version=OPENAI_DEPLOYMENT_VERSION,
                  openai_api_key=OPENAI_API_KEY,
                  openai_api_type="azure")
        # Initialize gpt-35-turbo and our embedding model
        #load the faiss vector store we saved into memory
        vectorStore = FAISS.load_local("./dbs/documentation/faiss_index", embeddings, allow_dangerous_deserialization = True,)
        #use the faiss vector store we saved to search the local document
        retriever = vectorStore.as_retriever(search_type="similarity", search_kwargs={"k":2})
        QUESTION_PROMPT = PromptTemplate.from_template("""Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.
        Chat History:
        {chat_history}
        Follow Up Input: {question}
        Standalone question:""")
        qa = ConversationalRetrievalChain.from_llm(llm=llm,
                                        retriever=retriever,
                                        condense_question_prompt=QUESTION_PROMPT,
                                        return_source_documents=True,
                                        verbose=False) 
        if 'qaValue' not in st.session_state:
            st.session_state['qaValue'] = qa
     
    with col1:           
           query = st.text_area("Enter some text", "Ask me anything...")
           st.markdown("")
           if query:
                # Perform an action (e.g., fetch an answer)
                with st.spinner('Fetching Answer...'): 
                    if 'qaValue' in st.session_state:                                         
                        qaActual=st.session_state['qaValue']
                        chatCompletion=[]
                        promptText=  "\n\n" + query +" in the above content"
                        result = qaActual({"question": promptText, "chat_history": chatCompletion})
                        st.markdown("<p style='color: green;font-weight:bold;'>"+result["answer"]+"\n </p>", unsafe_allow_html=True)                              
if __name__ == '__main__':
    print(f'------------------------------------------------------')
    main()
