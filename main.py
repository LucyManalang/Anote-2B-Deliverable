from ingestion.text_ingest import ingest_text
from indexing.index_pipeline import IndexPipeline
from query_fusion.query_engine import QueryEngine

def main():
    print("➡ Step 1: Ingesting…")
    docs = ingest_text("data/test.md")
    print(f"Ingested {len(docs)} chunks")

    print("➡ Step 2: Indexing…")
    pipeline = IndexPipeline()
    pipeline.index_documents(docs)

    print("➡ Step 3: Querying…")
    engine = QueryEngine(
        pipeline,
        llm_backend="ollama",
        model="llama3.2",
    )
    
    response = engine.ask("What animal appears in the document?")
    print("\n=== ANSWER ===")
    print(response["answer"])

    print("\n=== CITATIONS ===")
    print(response["citations"])

if __name__ == "__main__":
    main()
