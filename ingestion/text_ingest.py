from typing import List, Dict
from types import SimpleNamespace

def _load_markdown(path: str) -> List[SimpleNamespace]:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    return [SimpleNamespace(page_content=text, metadata={"source": path})]

def _load_pdf(path: str) -> List[SimpleNamespace]:
    import importlib
    PdfReader = None
    for mod_name in ("pypdf", "PyPDF2"):
        try:
            mod = importlib.import_module(mod_name)
        except Exception:
            continue
        # prefer PdfReader, but accept older PdfFileReader if present
        if hasattr(mod, "PdfReader"):
            PdfReader = getattr(mod, "PdfReader")
            break
        if hasattr(mod, "PdfFileReader"):
            PdfReader = getattr(mod, "PdfFileReader")
            break
    if PdfReader is None:
        raise ImportError(
            "pypdf or PyPDF2 is required to load PDFs. Install with: pip install pypdf"
        )

    reader = PdfReader(path)
    docs = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        docs.append(SimpleNamespace(page_content=text, metadata={"source": path, "page": i + 1}))
    return docs

class SimpleSplitter:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_documents(self, docs: List[SimpleNamespace]) -> List[SimpleNamespace]:
        out: List[SimpleNamespace] = []
        for doc in docs:
            text = doc.page_content or ""
            start = 0
            L = len(text)
            if L == 0:
                out.append(SimpleNamespace(page_content="", metadata=dict(doc.metadata)))
                continue
            while start < L:
                end = start + self.chunk_size
                chunk_text = text[start:end]
                out.append(SimpleNamespace(page_content=chunk_text, metadata=dict(doc.metadata)))
                start = end - self.chunk_overlap
                if start < 0:
                    start = 0
                if start >= L:
                    break
        return out

def ingest_text(path: str) -> List[Dict]:
    """
    Parse and chunk Markdown or PDF into text segments.
    Returns a list of {text, metadata} dictionaries.
    """
    if path.endswith(".md"):
        docs = _load_markdown(path)
    elif path.endswith(".pdf"):
        docs = _load_pdf(path)
    else:
        raise ValueError("Unsupported file type. Must be .md or .pdf")

    splitter = SimpleSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    return [{"text": c.page_content, "metadata": c.metadata} for c in chunks]

if __name__ == "__main__":
    from pprint import pprint
    pprint(ingest_text("README.md")[:2])
