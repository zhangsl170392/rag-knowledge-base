
# RAG 知识底座（基于 LangChain + Ragas + Ollama）

这是一个本地化部署的 RAG（检索增强生成）知识库与自动化评估系统。项目集成了文档解析、向量检索、基于本地大模型的生成，以及使用 Ragas 框架进行的严格量化评估。

## 一、核心特性

1. **全本地化运行**：依赖 Ollama 本地部署 LLM 与 Embedding 模型，数据不出域。
2. **自动化评估**：原生集成 Ragas，采用四大核心指标（忠实度、答案相关性、上下文精确率、上下文召回率）验证 RAG 效果。
3. **标准化检索接口**：提供统一的 retrieve() 接口契约，方便上层 Agent 或应用无缝调用。
4. **拒答机制**：当检索得分低于阈值时，接口返回空列表，有效防止模型幻觉。

## 二、环境准备
### 1. 基础环境
   - Python >= 3.10
   - 已安装并启动 Ollama (https://ollama.com/)

### 2. 拉取本地模型
请根据你的 config.py 中的配置，提前拉取所需的模型：
```bash
# 拉取 LLM 模型
ollama pull qwen2.5:7b

# 拉取 Embedding 模型
ollama pull nomic-embed-text
```

### 3. 安装依赖

> ⚠️ **极其重要的避坑指南**:
> 本项目的 requirements.txt 中强制锁定了 openai==1.109.1 和 litellm==1.75.3。
> 原因在于新版 litellm 强制要求 openai>=2.x，这会与 instructor、langchain-openai 等库产生严重冲突（导致 ResolutionImpossible 报错）。
> 请勿随意升级或删除这两个包的版本号。
   
在终端执行以下命令：

```bash
python -m venv .venv

# Windows 激活环境
.venv\Scripts\activate

# Mac/Linux 激活环境
source .venv/bin/activate

pip install -r requirements.txt
```

## 三、快速开始
### 第一步：配置环境变量
在项目根目录创建 .env 文件，并配置本地 Ollama 服务地址（默认如下）：
```env
OLLAMA_BASE_URL=http://localhost:11434
```

### 第二步：构建向量索引
将需要检索的文档（PDF / TXT / Markdown）放入 data/raw/ 目录，然后运行：
```bash
python src/rag/ingest.py
```
（该步骤会自动切分文档，注入元数据 source 和 chunk_id，并存入 ChromaDB。）

### 第三步：运行自动化评估
确保 eval/eval_set.json（评估数据集）已经准备好，然后执行：
```bash
python src/rag/evaluate.py
```
评估完成后，你将在 eval/results/ 目录下得到：
- summary.json：整体平均指标汇总。
- details.jsonl：每一道题的详细检索上下文、生成回答与单项得分。


## 四、核心接口契约：retrieve()

如果你的上层应用或 Agent 需要调用知识库，请使用以下标准接口：

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
**拒答逻辑**：如果所有返回结果都低于 score_threshold（默认 0.3），接口将直接返回空列表 []，上层应用应据此触发降级逻辑（如回答“我不知道”）。


## 五、评估指标说明

项目使用 Ragas 0.4.3 版本进行量化评估，包含以下四个维度：
1. **Faithfulness (忠实度)**：回答是否完全基于检索到的上下文（防幻觉）。
2. **Answer Relevancy (答案相关性)**：生成的回答是否直接解决了用户的问题。
3. **Context Precision (上下文精确率)**：检索出的文本块中，相关信息的排位是否靠前。
4. **Context Recall (上下文召回率)**：标准答案中的信息是否被检索系统完整找到。


## 六、常见问题排查（踩坑记录）

1. **报错：`Collections metrics only support modern embeddings`**
   - **解决**：请确保 `evaluate.py` 中使用了 `embedding_factory("litellm", ...)`，而不是 Ragas 旧版的 `LangchainEmbeddingsWrapper`。

2. **报错：Cannot install ... because these package versions have conflicting dependencies**
   - **解决**：请检查 requirements.txt 中的 litellm 是否被意外升级，必须保持在 1.75.3。

3. **ChromaDB 报错：`tenant default_tenant`**
   - **解决**：若从旧版本升级，旧格式的持久化数据（.sqlite3）可能不兼容 ChromaDB 1.5+。请清理旧的 chroma_db 目录，并重新运行 ingest.py 建立索引。

4. **LiteLLM 报错：`ReadTimeout fetching https://raw.githubusercontent.com...`**
   - **解决**：这是 LiteLLM 在拉取模型价格表时网络超时。在 evaluate.py 最顶部添加 os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True" 即可消除该警告。