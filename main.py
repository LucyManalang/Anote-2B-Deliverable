from ingestion.text_ingest import ingest_text
from ingestion.image_ingest import ingest_images
from indexing.index_pipeline import IndexPipeline
from query_fusion.query_engine import QueryEngine

def main():
    print("➡ Step 1: Ingesting…")
    text_docs = ingest_text("data/test.md")
    image_docs = ingest_images([
    "data/image_test_1.jpg",
    "data/image_test_2.jpg"])

    docs = text_docs + image_docs

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
    
    response = engine.ask("What animal appears in the images and what is the main theme of the text document?")
    print("\n=== ANSWER ===")
    print(response["answer"])

    print("\n=== CITATIONS ===")
    print(response["citations"])

if __name__ == "__main__":
    main()
