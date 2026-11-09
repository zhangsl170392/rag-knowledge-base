import argparse
from pathlib import Path
from dotenv import load_dotenv

# LangChain 相关组件
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import RAW_DATA_DIR, VECTOR_STORE_DIR, CHUNK_SIZE, CHUNK_OVERLAP, COLLECTION_NAME
from langchain_chroma import Chroma
from embed import get_embeddings

# 加载 .env 环境变量
load_dotenv()

# 默认配置（可以通过环境变量覆盖）
BASE_DIR = Path(__file__).resolve().parent.parent.parent


def load_documents(directory: str):
    """
    遍历目录，加载所有支持的文档。
    支持 .pdf, .txt, .md
    """
    docs = []
    dir_path = Path(directory)

    if not dir_path.exists():
        raise FileNotFoundError(f"数据目录不存在: {directory}")

    print(f"正在扫描目录: {directory}")

    for filepath in dir_path.rglob("*"):
        if filepath.is_file():
            ext = filepath.suffix.lower()
            try:
                if ext == ".pdf":
                    loader = PyPDFLoader(str(filepath))
                elif ext == ".txt":
                    loader = TextLoader(str(filepath), encoding="utf-8")
                elif ext in [".md", ".markdown"]:
                    loader = UnstructuredMarkdownLoader(str(filepath))
                else:
                    print(f"  跳过不支持的文件格式: {filepath.name}")
                    continue

                # 加载文档并记录来源路径 (Agent 溯源的关键)
                loaded_docs = loader.load()
                for doc in loaded_docs:
                    doc.metadata["source"] = str(filepath.name)  # 仅保留文件名，保护本地绝对路径隐私

                docs.extend(loaded_docs)
                print(f"  成功加载: {filepath.name}")
            except Exception as e:
                print(f"  加载 {filepath.name} 失败: {e}")

    print(f"共加载了 {len(docs)} 页/个文档。")
    return docs


def split_documents(docs):
    """
    将长文档切分为适合 Embedding 的 Chunk。
    使用 RecursiveCharacterTextSplitter，按段落、句号、换行符依次递归切分。
    """
    print(f"正在进行文本切分 (Chunk Size: {CHUNK_SIZE}, Overlap: {CHUNK_OVERLAP})...")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        is_separator_regex=False,
    )

    chunks = text_splitter.split_documents(docs)
    print(f"切分完成，共生成 {len(chunks)} 个 Chunk。")

    # 给每个 chunk 加上唯一的 ID 标记（可选，Chroma 会自动生成 UUID）
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = f"chunk_{i:04d}"

    return chunks


def ingest_data():
    """
    执行完整的摄入流水线：加载 -> 切分 -> 向量化 -> 存入 ChromaDB
    """
    # 1. 加载文档
    docs = load_documents(RAW_DATA_DIR)
    if not docs:
        print("没有找到文档，退出。")
        return

    # 2. 切分文档
    chunks = split_documents(docs)

    # 3. 获取向量模型单例

    embeddings = get_embeddings()

    print(f"正在将向量写入本地数据库: {VECTOR_STORE_DIR}")
    # 注意：这会在本地创建一个持久化的文件夹
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=VECTOR_STORE_DIR,
        collection_name=COLLECTION_NAME
    )

    print(f"✅ 摄入完成！知识底座已更新。")
    print(f"💡 向量库位置: {VECTOR_STORE_DIR}")


if __name__ == "__main__":
    # 支持命令行传参，如：python -m rag.ingest --dir data/raw
    parser = argparse.ArgumentParser(description="Agent 知识底座数据摄入脚本")
    parser.add_argument("--dir", type=str, default=RAW_DATA_DIR, help="原始文档所在目录")
    args = parser.parse_args()

    RAW_DATA_DIR = args.dir
    ingest_data()