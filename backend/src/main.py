from strategies.adaptive_retrieval import AdaptiveRAG

if __name__ == "__main__":
    rag_system = AdaptiveRAG()

    queries = [
        "Tell me about the Cold Bore Technology anything you know",
    ]

    for query in queries:
        print(f"Query: {query}")
        rag_system.answer(query)
