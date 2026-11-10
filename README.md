
# RAG 知识底座（基于 LangChain + Ragas + Ollama）

**这是 Agent 作品集的知识底座**：一个可被检索 Agent 调用的本地化知识库。它包含文档摄入、切分、向量索引、检索接口和 RAGAS 评估链路。它不直接面向终端用户，而是为上层 Agent 提供“可引用的知识”与标准化的检索接口。

## 一、数据来源与结构

本项目的知识库面向技术文档领域，支持 PDF / TXT / Markdown 格式。
- 原始数据存放于 `data/raw/`。
- 切分后的文本块会注入元数据（`chunk_id`、`source`、`text`），并存入本地的 ChromaDB。

## 二、核心接口契约：`retrieve()`
上层应用或 Agent 需要调用知识库时，请使用以下标准接口：
```python
from src.rag.retrieve import retrieve

# 返回结果按匹配得分降序排列
hits = retrieve(
    query="URL命名应该用下划线还是短横线？",
    top_k=3,
    score_threshold=0.3
)
```
返回的数据结构示例：
```json
[
  {
    "text": "切分后的文本块内容...",
    "source": "coding_guidelines.md",
    "score": 0.8123,
    "metadata": {"source": "coding_guidelines.md", "chunk_id": "chunk_0012"}
  }
]
```

## 三、检索参数与拒答策略

- 向量库：ChromaDB（支持本地持久化，无需额外服务，适合轻量底座）。
- Embedding 模型：使用 Ollama 本地部署的 nomic-embed-text。
- 拒答策略（防幻觉）：如果所有返回结果都低于 score_threshold（默认 0.3），接口将直接返回空列表 []。上层 Agent 拿到空列表后，应触发降级逻辑（如直接回答“我无法从知识库中找到相关信息”）。

## 四、评估证据（RAGAS 0.4.3）
项目原生集成 Ragas 框架进行量化评估，证明底座经过验收。评估集包含 50 道基于真实场景的 QA 对（位于 eval/eval_set.json）。
当前基线指标如下（详见 eval/results/summary.json）：
- Faithfulness (忠实度)：0.90 （几乎无幻觉）
- Answer Relevancy (答案相关性)：0.85
- Context Precision (上下文精确率)：0.79
- Context Recall (上下文召回率)：0.83

## 五、已知限制（Known Limitations）
- 暂不支持扫描版 PDF（图片型）的 OCR 解析。
- 对于包含复杂跨页表格的 PDF，切分效果较差，可能会导致上下文丢失。
- 目前仅支持中文检索，跨语言问答效果未经验证。

## 六、快速开始与目录结构
在终端执行以下命令：

```bash
# 1. 环境准备
python -m venv .venv

# Windows 激活环境
.venv\Scripts\activate

# Mac/Linux 激活环境
source .venv/bin/activate

pip install -r requirements.txt

# 2. 配置环境变量
cp .env.example .env
# 编辑 .env 填入你的 OLLAMA_BASE_URL

# 3. 构建向量索引（将文档放入 data/raw/ 后运行）
python src/rag/ingest.py

# 4. 运行自动化评估
python src/rag/evaluate.py
```
目录结构：
``` text
rag-knowledge-base/
├── data/
│   ├── raw/          # 存放原始 PDF/MD 文件
│   └── processed/    # ChromaDB 持久化目录（已加入 gitignore）
├── eval/             # 评估集与评估结果
├── notebooks/        # Demo 演示代码（demo.ipynb）
└── src/              # 核心源码
    └── rag/          # ingest、retrieve、evaluate 等核心逻辑
```

## 七、常见问题排查（踩坑记录）
> ⚠️ 极其重要的依赖避坑：本项目强制锁定了 openai==1.109.1 和 litellm==1.75.3。新版 litellm 会导致依赖冲突，请勿随意升级。

1. **报错：`Collections metrics only support modern embeddings`**
   - **解决**：请确保 `evaluate.py` 中使用了 `embedding_factory("litellm", ...)`，而不是 Ragas 旧版的 `LangchainEmbeddingsWrapper`。

2. **ChromaDB 报错：`tenant default_tenant`**
   - **解决**：若从旧版本升级，旧格式的持久化数据（.sqlite3）可能不兼容 ChromaDB 1.5+。请清理旧的 chroma_db 目录，并重新运行 ingest.py 建立索引。

3. **LiteLLM 报错：`ReadTimeout fetching https://raw.githubusercontent.com...`**
   - **解决**：这是 LiteLLM 在拉取模型价格表时网络超时。在 evaluate.py 最顶部添加 os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True" 即可消除该警告。