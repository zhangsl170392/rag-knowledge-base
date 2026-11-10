import os
from pathlib import Path
from dotenv import load_dotenv

# 加载 .env 文件
load_dotenv()

# 动态定位项目根目录
BASE_DIR = Path(__file__).resolve().parent.parent.parent

def _env_path(key: str, default: Path) -> Path:
    """读取路径配置：相对路径统一基于项目根目录解析。"""
    value = os.getenv(key)
    if not value:
        return default
    path = Path(value)
    return path if path.is_absolute() else BASE_DIR / path

# 数据路径配置
RAW_DATA_DIR = _env_path("RAW_DATA_DIR", BASE_DIR / "data" / "raw")
VECTOR_STORE_DIR = _env_path("VECTOR_STORE_DIR", BASE_DIR / "data" / "processed" / "chroma_db")

# 文本切分配置
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))

# 模型配置 (Ollama)
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "qwen3-embedding:0.6b")
LLM_MODEL = os.getenv("LLM_MODEL", "modelscope.cn/Qwen/Qwen2.5-7B-Instruct-GGUF:latest") # 用于评估时生成回答

# 向量库集合名称
COLLECTION_NAME = "knowledge_base"