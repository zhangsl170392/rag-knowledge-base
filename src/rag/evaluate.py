import asyncio
import json
import os
import pandas as pd

from openai import AsyncOpenAI
from ragas import experiment, Dataset
# from datasets import Dataset
from ragas.metrics.collections import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall
)
from ragas.llms import llm_factory

os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
from ragas.embeddings.base import embedding_factory
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

from config import LLM_MODEL, OLLAMA_BASE_URL, BASE_DIR, EMBEDDING_MODEL
from retrieve import retrieve
from embed import  get_embeddings

EVAL_SET_PATH = BASE_DIR / "eval" / "eval_set.jsonl"
RESULTS_DIR = BASE_DIR / "eval" / "results"

"""1. 使用本地 Ollama LLM 基于检索到的 context 生成回答"""
llm = ChatOllama(model=LLM_MODEL, base_url=OLLAMA_BASE_URL)

def generate_answer(query: str, contexts: list[str]) -> str:

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "你是一个严谨的助手。请严格基于以下提供的上下文回答问题。如果上下文中没有答案，请直接说'不知道'。\n上下文：\n{context}"),
        ("human", "{question}")
    ])
    chain = prompt | llm
    response = chain.invoke({"context": "\n\n".join(contexts), "question": query})
    return response.content

# ---------- 2. 用于 Ragas 评估的 LLM（v0.4 需要 llm_factory）----------
# 注意：ragas 0.4 的 collections 指标要求 InstructorBaseRagasLLM，
# 不能直接传入 LangChain 的 ChatOllama。
_eval_client = AsyncOpenAI(
    base_url=f"{OLLAMA_BASE_URL}/v1",  # Ollama 的 OpenAI 兼容端点
    api_key="ollama",                   # Ollama 不校验 key，但客户端要求非空
)
eval_llm = llm_factory(LLM_MODEL, provider="openai", client=_eval_client)

#   关键修复：包装 OllamaEmbeddings 单例，AnswerRelevancy 需要的是实例而不是模型名字符串
#   用 get_embeddings() 保证与 ingest / retrieve 用同一模型
ragas_embeddings = embedding_factory(
    "litellm",
    model=f"ollama/{EMBEDDING_MODEL}",
    api_base=OLLAMA_BASE_URL
)

# ---------- 3. 实验函数（只定义一次！）----------
@experiment()
async def run_evaluation_experiment(row):
    question = row["user_input"]
    hits = await asyncio.to_thread(retrieve, question, top_k=3, score_threshold=0.3)
    contexts = [hit["text"] for hit in hits]
    response = await asyncio.to_thread(generate_answer, question, contexts)

    faithfulness_score = await Faithfulness(llm = eval_llm).ascore(
        user_input=question,
        response=response,
        retrieved_contexts=contexts
    )

    answer_relevancy_score = await AnswerRelevancy(llm= eval_llm, embeddings=ragas_embeddings).ascore(
        user_input=question,
        response=response,
    )

    context_precision_score = await ContextPrecision(llm= eval_llm).ascore(
        user_input=question,
        reference=row["reference"],
        retrieved_contexts=contexts
    )

    context_recall_score = await ContextRecall(llm= eval_llm).ascore(
        user_input=question,
        reference=row["reference"],
        retrieved_contexts=contexts
    )

    return {
        "user_input": question,
        "reference": row["reference"],
        "response": response,
        "contexts": contexts,
        "faithfulness": faithfulness_score.value,
        "answer_relevancy": answer_relevancy_score.value,
        "context_precision": context_precision_score.value,
        "context_recall": context_recall_score.value
    }

# ---------- 4. 同步入口（负责加载数据、运行异步实验、保存结果）----------
def main():
    if not EVAL_SET_PATH.exists():
        print(f"❌ 找不到评估集文件: {EVAL_SET_PATH}")
        return

    with open(EVAL_SET_PATH, "r", encoding="utf-8") as f:
        eval_data = json.load(f)

    # Ragas v0.4 的 Dataset 字段名：
    # question -> user_input, ground_truth -> reference
    # response / contexts 在实验函数内部生成，但 Dataset 初始化时可以先留空
    dataset = Dataset(
        name="rag_eval_dataset",
        backend="local/jsonl",
        root_dir=str(RESULTS_DIR),
        data=[
        {
            "user_input": item["question"],
            "reference": item["ground_truth"]
        }
        for item in eval_data
        ]
    )

    print(f"已加载 {len(eval_data)} 条评估数据，开始运行实验...")

    # 用 asyncio.run 执行异步实验
    results = asyncio.run(run_evaluation_experiment.arun(dataset, name="rag_eval_run"))

    # 聚合指标并写盘
    # csv_path = os.path.join("experiments", "rag_eval_run.csv")
    # df = pd.read_csv(csv_path)
    df = pd.DataFrame(results)
    metric_cols = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]

    # 把原 json 的 id / source_section 回填，便于逐条 trace
    id_map = {item["question"]: item.get("id", "") for item in eval_data}
    sec_map = {item["question"]: item.get("source_section", "") for item in eval_data}
    df["eval_id"] = df["user_input"].map(id_map)
    df["source_section"] = df["user_input"].map(sec_map)

    summary = {
        "n_samples": len(df),
        "llm_model": LLM_MODEL,
        "embedding_model": EMBEDDING_MODEL,
        "top_k": 3,
        "score_threshold": 0.3,
        "metrics": {c: round(float(df[c].mean()), 4) for c in metric_cols},
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = RESULTS_DIR / "summary.json"
    details_path = RESULTS_DIR / "details.jsonl"

    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with open(details_path, "w", encoding="utf-8") as f:
        for row in df.to_dict(orient="records"):
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")

    print("\n✅ 评估完成")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\n汇总指标: {summary_path}")
    print(f"逐条明细: {details_path}")

if __name__ == "__main__":
    main()