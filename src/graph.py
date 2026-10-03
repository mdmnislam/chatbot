import os
from typing import TypedDict, Annotated
from langgraph.graph import add_messages, StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain.schema import Document
from src.helper import text_embeddings
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
load_dotenv()

memory = MemorySaver()
index_name = "medical-chatbot"
embeddings = text_embeddings()

vectorsearch = PineconeVectorStore.from_existing_index(
    embedding=embeddings,
    index_name=index_name
)
retriever = vectorsearch.as_retriever(search_type="similarity",  search_kwargs={"k": 3})

class BasicAgent(TypedDict):
    messages: Annotated[list, add_messages]
    documents: list[Document]
    on_topic: str
    question: str

def query_class(state: BasicAgent):
    query = state["messages"][-1].content
    
    message ="""You are a classifier that determines if a query is on medical 
                related topics or not. If query is on medical related topics, 
                respond with 'Yes', otherwise respond with 'No'."""
    
    prompt = [("system", message), 
              ("human", query)]
    
    llm = ChatOpenAI(model="gpt-4o",
                    api_key=os.environ.get("OPENAI_API_KEY")
        )
    result = llm.invoke(prompt)
    
    return {"messages": [result], 
            "on_topic": result.content.strip(),
            "question": query}

def topic_router(state: BasicAgent):
    on_topic = state["on_topic"]
    if on_topic.lower() == "yes":
        return "Yes"
    else:
        return "No"
    
def off_topic(state: BasicAgent):
    return {"messages": [HumanMessage(content="Sorry, I can only answer medical related queries.")]}
    
def on_topic_retriever(state: BasicAgent, retriever=retriever):
    question = state["question"]
    documents = retriever.invoke(question)
    
    return {"documents": documents}

def generator(state: BasicAgent):
    question = state["question"]
    # documents = state["documents"]
    documents = state.get("documents", [])
    if not documents:
        print("⚠️ Warning: No documents found in state context!")
    
    context = "\n".join([doc.page_content for doc in documents])
    
    message = f"""You are a medical chatbot that provides information based on the 
                  provided context. Answer the following question based on the context 
                  below. If the answer is not present in the context, respond with 
                  'I don't know'.
                  
                  Context: {context}
                  Question: {question}"""
    
    prompt = [("system", message)]
    
    llm = ChatOpenAI(model="gpt-4o",
                    api_key=os.environ.get("OPENAI_API_KEY")
        )
    result = llm.invoke(prompt)
    
    return {"messages": [result], "on_topic": result.content.strip(), "documents": documents}

def graph_builder():   
    graph = StateGraph(BasicAgent)
    graph.add_node("query_class", query_class)
    graph.set_entry_point("query_class")
    graph.add_node("off_topic", off_topic)
    graph.add_node("on_topic_retriever", on_topic_retriever)
    graph.add_node("generator", generator)
    graph.add_conditional_edges("query_class", topic_router,
                                {"Yes": "on_topic_retriever", "No": "off_topic"})
    graph.add_edge("off_topic", END)
    graph.add_edge("on_topic_retriever", "generator")
    graph.add_edge("generator", END)

    app = graph.compile(checkpointer=memory)
    
    return app

