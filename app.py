import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA

# 1. SETUP PAGE - This creates your two sections
st.set_page_config(layout="wide", page_title="By-Law AI")

@st.cache_resource
def init_bot():
    # Load the fast folder you uploaded to GitHub
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # This reads your 'faiss_index' folder directly
    db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    
    # Connect to Groq (REPLACE THE KEY BELOW)
    llm = ChatGroq(
        groq_api_key="xai-9kdHzBm5MHwrv1CDRYz9r5e6gNTxaTNdKmBFsaNfNevGphPdZv36ykeDlibHgHYCy4WneCQecQuSWuIz", 
        model_name="llama3-70b-8192",
        temperature=0
    )
    return RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=db.as_retriever())

# Initialize the AI
bot = init_bot()

# 2. TWO SECTIONS (Question & Answer)
st.title("🏗️ Building Law Expert")
st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.header("Ask Your Question")
    user_query = st.text_area("Example: What is the minimum front setback for a residential building?", height=200)
    ask_button = st.button("Get Answer")

with col2:
    st.header("Legal Answer")
    if ask_button and user_query:
        with st.spinner("Analyzing 300 pages of laws..."):
            response = bot.invoke(user_query)
            st.success(response["result"])
