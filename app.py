import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA

# 1. SETUP PAGE
st.set_page_config(layout="wide", page_title="AI Law Assistant")

# 2. LOGIN SYSTEM
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

if not st.session_state["logged_in"]:
    st.title("🔐 Architect Access Only")
    user = st.text_input("Enter Login ID")
    pw = st.text_input("Enter Password", type="password")
    if st.button("Login"):
        if user == "architect1" and pw == "pass123": # Change these to sell!
            st.session_state["logged_in"] = True
            st.rerun()
        else:
            st.error("Wrong ID/Password")
    st.stop()

# 3. CHATBOT ENGINE
@st.cache_resource
def init_bot():
    loader = PyPDFLoader("laws.pdf")
    docs = loader.load_and_split()
    # Replace the text below with your Hugging Face Token from Step 1
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = FAISS.from_documents(docs, embeddings)
    # Replace 'YOUR_GROQ_KEY' with the key from Step 1
    llm = ChatGroq(groq_api_key="xai-9kdHzBm5MHwrv1CDRYz9r5e6gNTxaTNdKmBFsaNfNevGphPdZv36ykeDlibHgHYCy4WneCQecQuSWuIz", model_name="llama3-8b-8192")
    return RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=db.as_retriever())

bot = init_bot()

# 4. TWO-SECTION LAYOUT
st.title("🏢 Building By-Law Expert")
left, right = st.columns(2)

with left:
    st.header("Ask Your Question")
    query = st.text_area("Example: What is the minimum setback for a 500sqm plot?", height=200)
    submit = st.button("Get Answer")

with right:
    st.header("AI Response")
    if submit and query:
        with st.spinner("Searching laws..."):
            response = bot.invoke(query)
            st.success(response["result"])
