import threading
from typing import List, Dict, Any

from config import VECTOR_STORE_DIR, COLLECTION_NAME
from langchain_chroma import Chroma
from embed import get_embeddings


# 全局单例：避免每次调用 retrieve 都重新加载模型和数据库
_vectorstore = None
_vectorstore_lock = threading.Lock()


def get_vectorstore() -> Chroma:
    """
    获取或初始化 Chroma 向量库连接（单例模式）。
    """
    global _vectorstore
    embeddings = get_embeddings()

    with _vectorstore_lock:
        if _vectorstore is None:
            _vectorstore = Chroma(
                persist_directory=VECTOR_STORE_DIR,
                embedding_function=embeddings,
                collection_name=COLLECTION_NAME,
                collection_metadata={"hnsw:space": "cosine"}
            )
    return _vectorstore


def retrieve(query: str, top_k: int = 3, score_threshold: float = 0.5) -> List[Dict[str, Any]]:
    """
    Agent 调用的核心检索接口。

    参数:
        query (str): 用户的查询语句。
        top_k (int): 返回的最大文本块数量。
        score_threshold (float): 相似度阈值（0~1），低于此分数的结果将被丢弃，防止 Agent 产生幻觉。

    返回:
        List[Dict]: 包含 text, source, score, metadata 的字典列表。
    """
    vectorstore = get_vectorstore()

    # 使用 relevance score（余弦相似度），分数越高越相似（1.0 为完全一致）
    # results = vectorstore.similarity_search_with_relevance_scores(
    results = vectorstore.similarity_search_with_score(
        query,
        k=top_k
    )

    formatted_results = []
    for doc, distance in results:
        # 2. 这里针对 L2 距离做过滤。L2 距离越大越不相关。
        # 对于 L2，距离小于 1.2 通常算相关（具体视你的 Embedding 模型而定）
        if distance > 1.2:
            continue
        # print(f"🐞 DEBUG - 原始距离: {distance:.4f} | 内容: {doc.page_content[:30]}...")

        # 3. 将距离转化为 0~1 的相似度（为了展示好看，直接用 1 - 距离/2 近似转换）
        # L2 距离通常在 0~2 之间
        score = max(0.0, min(1.0, 1.0 - (distance / 2.0)))

        formatted_results.append({
            "text": doc.page_content,
            "source": doc.metadata.get("source", "未知来源"),
            "score": round(score, 4),
            "metadata": doc.metadata
        })

    return formatted_results


if __name__ == "__main__":
    # 简单的命令行测试
    import sys

    print("=" * 50)
    print("Agent 知识底座检索测试")
    print("=" * 50)

    # 如果你在终端传了参数，就用参数；否则用默认测试查询
    test_query = sys.argv[1] if len(sys.argv) > 1 else "URL命名应该用下划线还是短横线？"
    print(f"查询: {test_query}\n")

    try:
        hits = retrieve(test_query, top_k=3, score_threshold=0.3)

        if not hits:
            print("没有找到相关的知识。")
        else:
            for i, hit in enumerate(hits, 1):
                print(f"[{i}] 来源: {hit['source']} | 相似度: {hit['score']}")
                print(f"    内容: {hit['text'][:100]}...\n")

    except Exception as e:
        print(f"❌ 检索失败: {e}")