from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain
import streamlit as st
from PIL import Image
import fitz  # PyMuPDF for PDF handling
import docx
# ---- Streamlit App ----
st.title("CuriosityBot :robot_face:")
# ---- Sidebar: Tone Selection & Parameters ----
with st.sidebar:
    st.markdown("## Inference Parameters")
    TEMPERATURE = st.slider("Temperature", 0.0, 1.0, 0.1, 0.1) # Temperature: Controls randomness in the response
    TOP_P = st.slider("Top-P", 0.0, 1.0, 0.9, 0.01) # Top-P: Limits selection to a probability threshold for the next word.
    TOP_K = st.slider("Top-K", 1, 500, 10, 5) # Top-K: Limits the selection to the top K most probable next words.
    MAX_TOKENS = st.slider("Max Token", 0, 1024, 512, 8) # Max Tokens: Restricts the maximum length of the response in tokens.
    MEMORY_WINDOW = st.slider("Memory Window", 0, 10, 3, 1) # Memory Window: Controls how many previous messages the bot can "remember" during the conversation.

# ---- Define LLM and Embeddings ----
hf_model = "mistralai/Mistral-7B-Instruct-v0.3"
llm = HuggingFaceEndpoint(repo_id=hf_model)
embedding_model = "sentence-transformers/all-MiniLM-l6-v2"
embeddings_folder = "/content/"
embeddings = HuggingFaceEmbeddings(model_name=embedding_model, cache_folder=embeddings_folder)
# Load Vector Database
vector_db = FAISS.load_local("/content/faiss_index", embeddings, allow_dangerous_deserialization=True)
retriever = vector_db.as_retriever(search_kwargs={"k": 2})  # Retrieve top 2 sources
# ---- Define Chatbot Prompt Based on Tone ----
# ---- Define Chatbot Tones ----
tones = {
    "Expert": {
        "description": "Provides precise, detailed, and informative answers in a professional tone.",
        "template": """You are an expert chatbot providing precise, detailed, and informative answers. Respond in a professional and knowledgeable tone.
Previous conversation:
{{chat_history}}
Context to answer question:
{context}
New human question: {input}
Response:"""
    },
    "Rude": {
        "description": "Gives no-nonsense, blunt, and direct answers with no tolerance for unnecessary details.",
        "template": """You are a blunt chatbot that gives no-nonsense, direct answers. Be rude and to the point, with no tolerance for unnecessary details.
Previous conversation:
{{chat_history}}
Context to answer question:
{context}
New human question: {input}
Response:"""
    },
    "Friendly": {
        "description": "Engages in a warm, conversational, and approachable manner while providing helpful answers.",
        "template": """You are a friendly and engaging chatbot. Respond in a warm, conversational, and helpful manner, making sure to keep things light and approachable.
Previous conversation:
{{chat_history}}
Context to answer question:
{context}
New human question: {input}
Response:"""
    },
    "Sarcastic": {
        "description": "Responds with a witty and sarcastic tone while still providing relevant information.",
        "template": """You are a sarcastic chatbot. Respond with witty and humorous remarks while still providing useful information.
Previous conversation:
{{chat_history}}
Context to answer question:
{context}
New human question: {input}
Response:"""
    }
}
# ---- Sidebar: Tone Selection ----
tone = st.sidebar.selectbox("Select Bot Tone:", list(tones.keys()))
# ---- Select Chatbot Template Based on Tone ----
selected_tone = tones[tone]
template = selected_tone["template"]
prompt = ChatPromptTemplate.from_messages([
    ("system", template),
   MessagesPlaceholder(variable_name="chat_history", n_messages=1), 
    ("human", "{input}"),
])

##"chat_history": st.session_state.messages,"chat_history": st.session_state.messages,
# ---- Initialize Chatbot ----
@st.cache_resource
def init_bot():
    doc_retriever = create_history_aware_retriever(llm, retriever, prompt)
    doc_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(doc_retriever, doc_chain)
rag_bot = init_bot()
# ---- Initialize Chat History ----
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.sources = []
# ---- File Upload Feature ----
st.markdown("## :open_file_folder: Upload a File")
uploaded_files = st.file_uploader("Upload any file", type=None, accept_multiple_files=True)
if uploaded_files:
    for uploaded_file in uploaded_files:
        st.markdown(f"### :file_folder: {uploaded_file.name}")
        st.write(f"**File Type:** {uploaded_file.type}")
        st.write(f"**Size:** {uploaded_file.size / 1024:.2f} KB")
        # ---- Process Different File Types ----
        if uploaded_file.type.startswith("image"):
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
        elif uploaded_file.type == "text/plain":
            text = uploaded_file.read().decode("utf-8")
            st.text_area("Text Content", text, height=200)
        elif uploaded_file.type == "application/pdf":
            pdf_reader = fitz.open(stream=uploaded_file.read(), filetype="pdf")
            pdf_text = "\n".join([page.get_text("text") for page in pdf_reader])
            st.text_area("PDF Content", pdf_text, height=200)
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            doc = docx.Document(uploaded_file)
            doc_text = "\n".join([p.text for p in doc.paragraphs])
            st.text_area("DOCX Content", doc_text, height=200)
        else:
            st.warning("Unsupported file type.")
st.success("File(s) uploaded successfully!") if uploaded_files else None

# ---- Display Chat History in Main Area ----
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(str(message["content"]))
# ---- Chat Input Section ----
if user_input := st.chat_input("Seeking Knowledge? Ask Me!"):
    # Display User Message
    st.chat_message("human").markdown(user_input)
    # Save User Message in Chat History
    st.session_state.messages.append({"role": "human", "content": user_input})
    # Show Spinner While Processing
    with st.spinner("Thinking..."):
        # Get Chatbot Response
        response_data = rag_bot.invoke({"input": user_input, "chat_history": st.session_state.messages, "context": retriever})
        response_text = response_data["answer"]
        # ---- Extract Citations ----
       # Cite its sources
   
        # Display Chatbot Response
        with st.chat_message("assistant"):
            st.markdown(response_text)
        # Save Assistant Response in Chat History
        st.session_state.messages.append({"role": "assistant", "content": response_text})
        st.session_state.sources.append(response_data["context"])
        #t.session_state.messages.append({"role": "assistant", "content": response_text})

if st.session_state.sources:
    if st.button ("Find source"):
        st.sidebar.markdown("# Sources")
        recent_sources = st.session_state.sources[-1]
        if len(recent_sources) == 0:
            st.sidebar.write ("No relevant sources found")
        else:
            for doc in recent_sources:
                source_name = doc.metadata["page"]
            content =doc.page_content
            st.sidebar.markdown(f"## Cite: \nPage: {source_name}, \n \nContent: \n{content}")
# ---- Display Chat History in Sidebar ----
with st.sidebar:
    st.markdown("## Chat History")
    for message in st.session_state.messages:
        st.markdown(f"**{message['role'].capitalize()}:** {message['content']}")

