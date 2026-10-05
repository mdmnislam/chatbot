from langchain_core.messages import HumanMessage
from src.graph import graph_builder

if __name__ == "__main__":
    app = graph_builder()
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit", "end"]:
            print("Exiting the application.")
            break
        response = app.invoke(
            {"messages": [HumanMessage(content=user_input)]},
            config={"configurable": {"thread_id": "1"}} # Required for MemorySaver
        )
        print("---------------------------------")
        print(response["messages"][-1].content)
        print("---------------------------------")
