from langchain_ollama import OllamaEmbeddings
from config import EMBEDDING_MODEL, OLLAMA_BASE_URL

_embedding_instance = None

def get_embeddings() -> OllamaEmbeddings:
    """
    获取全局唯一的 Ollama Embedding 实例。
    保证 ingest (写入) 和 retrieve (读取) 使用的是完全相同的模型。
    """
    global _embedding_instance
    if _embedding_instance is None:
        _embedding_instance = OllamaEmbeddings(
            model=EMBEDDING_MODEL,
            base_url=OLLAMA_BASE_URL
        )
    return _embedding_instance