import os
import streamlit as st
from dotenv import load_dotenv, find_dotenv
from urllib.parse import urlparse, parse_qs
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_core.tools import tool
from langchain.tools import Tool
from duckduckgo_search import DDGS
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage
from langchain.memory import ConversationBufferWindowMemory
from langchain_openai import ChatOpenAI

_ = load_dotenv(find_dotenv())

OPENAI_API_KEY  = os.getenv('OPENAI_API_KEY') 
LANGCHAIN_API_KEY = os.getenv('LANGCHAIN_API_KEY')
os.environ["LANGCHAIN_TRACING_V2"] = "true"
embedding_model = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

faiss_index_path = r"C:\Users\tiago\OneDrive\Ambiente de Trabalho\Last Project\parent-helper\faiss_youtube_index"
#vectorstore = FAISS.from_documents(split_docs, embedding_model)
vectorstore = FAISS.load_local(faiss_index_path, embeddings=embedding_model, allow_dangerous_deserialization=True)


# chat completion llm
llm = ChatOpenAI(
    openai_api_key=OPENAI_API_KEY,
    model_name='gpt-4o-mini',  # Ensure this model is supported
    temperature=0.0
)

# conversational memory
memory = ConversationBufferWindowMemory(
    memory_key='chat_history',
    k=5,
    return_messages=True
)

from langchain.chains import RetrievalQA
# Create a retriever from the loaded FAISS vector store
faiss_retriever = vectorstore.as_retriever(
    search_type="similarity", # Or "mmr", etc.
    search_kwargs={'k': 5}     # Number of documents to retrieve
)

# Initialize the QA chain with the FAISS retriever
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff", # Or "map_reduce", "refine", "map_rerank"
    retriever=faiss_retriever,
    return_source_documents=True # Optional: To see which chunks were retrieved
)

@tool
def calculator(expression: str) -> str:
    """Safely evaluate a basic school-level math expression like '2 + 3 * 4'."""
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return f"Error: {e}"

qa_tool = Tool(
    name="DocumentSearch",
    func=qa_chain.invoke,  # 'qa' is the RetrievalQA object 
    description="Use this tool to answer school-level questions about science, history or other topics."
)

@tool
def image_search(query: str) -> str:
    """Search for a safe, educational image to help explain the topic visually."""
    try:
        with DDGS() as ddgs:
            results = ddgs.images(query, max_results=3, safesearch="moderate")
            if results:
                return f"Here are some images that might help explain '{query}': {results[0]['image']}, {results[1]['image']}"
            return "No relevant image found."
    except Exception as e:
        return f"Image search failed: {str(e)}"

tools = [calculator, qa_tool, image_search]

custom_prompt = ChatPromptTemplate.from_messages([
    ("system", 
     "You are a warm, friendly AI tutor designed to assist parents in helping their children with schoolwork.\n\n"
     "Your job is to guide the parent with a clear, simple, step-by-step explanation of concepts, so they can teach their child effectively. "
     "After walking through the explanation and tool usage, you should always reorganize the reasoning steps and bullet points, to format the explanation to help the parents, and state the short answer in the end.\n\n"

     "Tools you should use:\n"
     "- A document search tool, use this tool first to find information about the topic.\n"
     "- A calculator tool for solving math problems, if is a math problem.\n"
     "- Lastly, use an image search tool to find helpful, educational images.\n\n"
     "Use all the tools for each question, if you can't use one tool, try to use the others. Before using it, say something like:\n"
     "'Let me check the educational materials...'\n"
     "'I’ll calculate this now...'\n"
     "'Let me find an image to help explain this visually...'\n\n"

     "Always simplify technical or complex terms so parents can explain them to younger children.\n\n"

     "If you can't find useful information, always say:\n"
     "'I couldn't find any direct information about {input}, but maybe this helps: ...' and then give your best guess or reasoning.\n\n"

     "Always reply in the user's language.\n\n"

     "Final Output Instructions:\n"
     "Must include three separate parts:\n"
     "- Include first a summarization, you should always reorganize the reasoning steps and bullet points, to format the explanation to help the parents explain to the kids.\n"
     "- Try to include an image if helpful.\n"
     "- State a shorter answer in the end, after the explanation.\n"
     ),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])

agent = create_openai_functions_agent(llm=llm, tools=tools, prompt=custom_prompt) 
agent_executor = AgentExecutor(agent=agent, tools=tools, memory=memory, verbose=True)

# 🔤 App title
st.title("📚 Parent Helper AI 📚")

# 🧠 Store chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# 📜 Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 📥 Chat input and response handling
if prompt := st.chat_input("Let me help you with your child's schoolwork"):
    # Show user message
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Run the agent (tool-powered LLM logic)
    response = agent_executor.invoke({"input": prompt})

    # Show AI response
    with st.chat_message("assistant"):
        st.markdown(response["output"])
    st.session_state.messages.append({"role": "assistant", "content": response["output"]})
