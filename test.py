import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory 
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain.llms import HuggingFaceHub
from supabase import create_client, Client

supabase_url = 'https://rquwntqrmfmwtzzlbjci.supabase.co'
supabase_key = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InJxdXdudHFybWZtd3R6emxiamNpIiwicm9sZSI6ImFub24iLCJpYXQiOjE3MDExMDYzNTEsImV4cCI6MjAxNjY4MjM1MX0.szFlkP1hTlddGoE8akJrt78fCjB1XVIhWF8ZrKCoxZw'

supabase: Client = create_client(supabase_url, supabase_key)


#all the text content will be returned here
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        #one pdf reader obj for each page
        pdf_reader = PdfReader(pdf)
        #then take content on each and append it to the text 
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

#now next we will take the pages in document and split it into chunks so that our LLM can process them
def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000, #process  the text in chunks of this size (default: 1000).
        chunk_overlap=200, #there will be a overlap of 200 tokens between the conseutive chunks
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

#now for these chunks we whave to create embeddings
#first understand what is embedding? - embedding is a process of creating vectors using deep learning. an embedding is the output of this process -- in other words, the vector that is created by a DL model for the purpose of similarity searches by that model.
#now what are embeddings? - in simple words embeddings are numerical representations of information, such as text, images, documents and audio. they represent each character as vector representation
# Vector embeddings are numerical representations of data that capture the meaning and relationships of words, phrases, and other data types.
# The distance between two vectors measures their relatedness. Small distances suggest high relatedness, and large distances suggest low relatedness. 
# In natural language processing (NLP), a word embedding is a representation of a word. The embedding is used in text analysis. The representation is a real-valued vector that encodes the meaning of the word. Words that are closer in the vector space are expected to be similar in meaning. 
#vector embedding that we will be using will take each chunk and we will use openai embeddings for that
        
#embeddings

        
#langchain supports a lot of vector stores
#One of the most common ways to store and search over unstructured data is to embed it and store the resulting embedding vectors, and then at query time to embed the unstructured query and retrieve the embedding vectors that are 'most similar' to the embedded query. 
#A vector store takes care of storing embedded data and performing vector search for you.
        
#here we will use the FAISS vector database
#here we have to pass chunks(documents) along with embeddings
def get_vectorstore(text_chunks):
    #embeddings = OpenAIEmbeddings()
    embeddings = HuggingFaceEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

#langchain allows you to have memory
#that means you can ask a question about your document and then you can ask a follow-up question too
#so the chatbot is going to remember the context of the question
#so all in all it will take the history of the conversation and return you the next converstion part
def get_conversation_chain(vectorstore):
    # llm = ChatOpenAI()
    llm = HuggingFaceHub(repo_id="HuggingFaceH4/zephyr-7b-beta", model_kwargs={"temperature":0.2, "max_length":10000})
    #initializing an instance of memory, it is called converstional buffer memory, there is also entity memory and other types of memory that langchain provies
    #converational retrieval chain allows us to chat with our vector store which has a memory in it
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(), #calling the vector store
        memory=memory
    )
    return conversation_chain  


def handle_userinput(user_question):
    #calling the conversation with the user question
    #st.session_state.conversation() has all the configuration of a vector space and from our memory this means it already knows our prvious questions
    #if we keep aksing questions its going to keep on remembering the question
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']
    #this will allow you to loop through the chat history with index and the content of the index
    for i, message in enumerate(st.session_state.chat_history):
        #since we are doing mod 2 so it will only take the odd numbers in the chat history
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)


def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs",
                       page_icon=":books:")
    st.write(css, unsafe_allow_html=True)
    #whenever using session state it should be initialized at the beginning of the application
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with multiple PDFs :books:")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                # get pdf text
                raw_text = get_pdf_text(pdf_docs)

                # get the text chunks
                text_chunks = get_text_chunks(raw_text)
                st.write(text_chunks)

                # create vector store
                vectorstore = get_vectorstore(text_chunks)

                # create conversation chain 
                # so every time a new converstion is striked this whole loop will iterate again there is a chance that it may re-initialize all the variables
                # to avoid that we will use the st.session_state, this will link the varibles to the current session state
                # and will avoid it from re-initializing
                st.session_state.conversation = get_conversation_chain(
                    vectorstore)

    # Feedback logic
    if st.session_state.chat_history:
        st.subheader("Feedback:")
        st.write("Was this answer helpful?")
        fed_thumbs_up, fed_thumbs_down = st.columns(2)
        # Add feedback buttons
        thumbs_up = fed_thumbs_up.button("Yes, it was helpful👍")
        thumbs_down = fed_thumbs_down.button("Provide a better answer👎")

        # Handle feedback
        if thumbs_up or thumbs_down:
            # Determine the feedback value
            feedback = 1 if thumbs_up else -1

            # Save feedback to Supabase database
            feedback_data = {"question": user_question, "answer": st.session_state.chat_history[-1].content, "feedback": feedback}
            # In this line, I assume that the last message in chat history is the response generated by the bot.
            st.write(feedback_data)  # You can remove this line, it's just for debugging purposes
            # Add code to save the feedback data to your Supabase database
            # supabase.table("feedback").insert([feedback_data]).execute() 
            feedback_table = supabase.table("feedback").insert([feedback_data]).execute()
            st.write(feedback_table)

            st.success("Feedback submitted successfully!")


if __name__ == '__main__':
    main()