import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
from sqlalchemy import create_engine
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from core.schema_formatter import get_full_schema
load_dotenv()

COLLECTION_NAME = "chinook_schema"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHROMA_PATH     = "./chroma_db"

class SchemaDiscoveryAgent:
    def __init__(self, engine):
        self.engine       = engine
        self.client       = None
        self.collection   = None
        self._ready       = False
        self.embedding_fn = SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)

    def initialize(self):
        if self._ready:
            return
        self.client     = chromadb.PersistentClient(path=CHROMA_PATH)
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME, embedding_function=self.embedding_fn,
            metadata={"description": "Chinook database table schemas"}
        )
        if self.collection.count() == 0:
            print("Schema cache empty — building from database...")
            self._build_cache()
        else:
            print(f"Schema cache loaded — {self.collection.count()} tables cached")
        self._ready = True

    def _build_cache(self):
        schema_dict = get_full_schema(self.engine)
        documents, ids, metadatas = [], [], []
        for table_name, create_table_str in schema_dict.items():
            documents.append(create_table_str)
            ids.append(table_name)
            metadatas.append({"table_name": table_name})
        self.collection.add(documents=documents, ids=ids, metadatas=metadatas)
        print(f"✅ Cached {len(documents)} table schemas in Chroma")

    def discover(self, question: str, top_k: int = 3) -> dict:
        if not self._ready:
            self.initialize()
        results = self.collection.query(
            query_texts=[question], n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
        matched_tables    = results["metadatas"][0]
        matched_documents = results["documents"][0]
        distances         = results["distances"][0]
        table_names       = [m["table_name"] for m in matched_tables]
        return {"schema": "\n".join(matched_documents), "tables_used": table_names, "distances": distances}

    def refresh(self):
        if self.collection:
            self.client.delete_collection(COLLECTION_NAME)
        self._ready = False
        self.initialize()

    def list_all_tables(self):
        if not self._ready:
            self.initialize()
        results = self.collection.get(include=["metadatas"])
        return [m["table_name"] for m in results["metadatas"]]