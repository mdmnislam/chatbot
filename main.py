from langchain_core.messages import HumanMessage
from src.graph import graph_builder

if __name__ == "__main__":
    app = graph_builder()
    response = app.invoke(
        {"messages": [HumanMessage(content="What is Acne?")]},
        config={"configurable": {"thread_id": "1"}} # Required for MemorySaver
    )
    print("---------------------------------")
    print(response["messages"][-1].content)
    print("---------------------------------")
