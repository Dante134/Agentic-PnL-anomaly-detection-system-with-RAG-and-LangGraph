# Agentic PnL Anomaly Detection System

A LangGraph agent that detects anomalous trade PnL, retrieves relevant
finance documentation via RAG, and generates plain-English explanations
using Claude. Built as a portfolio project for ML/Agentic AI roles.

<img width="1440" height="1560" alt="image" src="https://github.com/user-attachments/assets/07efc8e0-636c-436a-9b01-b9ffbde9f71a" />
<img width="1440" height="1560" alt="image" src="https://github.com/user-attachments/assets/e01cff29-bfc3-4320-9f13-144d29ff4b60" />

---

## What it does

1. **Ingests** synthetic trade data (swap for real data later)
2. **Detects** anomalies using z-score + domain rule-based detection
3. **Retrieves** relevant finance docs via ChromaDB vector search (RAG)
4. **Explains** each anomaly in plain English using Claude

---

## Architecture

```
AgentState (shared state dict)
     │
     ▼
[ingest] → load trade records
     │
     ▼
[detect] → z-score + rule-based flagging
     │
     ▼
[retrieve] → ChromaDB semantic search (RAG)
     │
     ▼
[explain] → Claude generates plain-English explanation
     │
     ├── more anomalies? → back to [retrieve]
     │
     └── done → print report + save JSON
```

LangGraph manages the loop automatically via a conditional edge.
The agent processes one anomaly at a time through retrieve → explain,
cycling until all flagged trades have been explained.

---

## Quickstart

```bash
# 1. Clone or download this folder

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set your Anthropic API key
export ANTHROPIC_API_KEY=sk-ant-your-key-here
# Windows: set ANTHROPIC_API_KEY=sk-ant-your-key-here

# 5. Run the agent
python pnl_agent.py
```

---

## Key concepts demonstrated

| Concept | Where in code |
|---|---|
| LangGraph StateGraph | `build_graph()` function |
| Typed state dict | `AgentState` TypedDict |
| Conditional looping edge | `should_continue()` function |
| RAG with ChromaDB | `build_rag_store()` + `node_retrieve()` |
| Z-score anomaly detection | `node_detect()` |
| LLM with grounded prompting | `node_explain()` |

---

## How to extend this

**Swap in real data:** Replace `node_ingest` to query a SQL database or
read a CSV instead of using the generator.

**Add more knowledge docs:** Drop `.txt` files in `knowledge_base/` and
update `build_rag_store()` to load them all.

**Use a persistent vector store:** Change `chromadb.Client()` to
`chromadb.PersistentClient(path="./chroma_db")` so embeddings survive
restarts.

**Add a human-in-the-loop node:** After `explain`, add a node that
prints the explanation and asks for user confirmation before logging —
a key pattern Morgan Stanley mentioned in their governance frameworks.

**Upgrade the detector:** Replace z-score with Isolation Forest or
an LSTM autoencoder for more sophisticated detection.

---

## Project structure

```
pnl_agent/
├── pnl_agent.py              ← main agent (LangGraph + RAG + Claude)
├── requirements.txt
├── README.md
├── data/
│   └── generate_trades.py    ← synthetic trade data generator
└── knowledge_base/
    └── pnl_concepts.txt      ← finance docs for RAG
```

---

## Expected output

```
📚 Building RAG knowledge base...
  Indexed 8 knowledge chunks.

🚀 Starting PnL anomaly detection agent...

[NODE 1: INGEST] Loading trade data...
  Loaded 72 trades across 9 trading days.

[NODE 2: DETECT] Running anomaly detection...
  Detected 9 anomalies from 72 trades.
  → TRD-1003 | Equity | PnL: $-712k | ['extreme_loss', 'z_score_outlier']
  ...

[NODE 3: RETRIEVE] RAG lookup for TRD-1003...
[NODE 4: EXPLAIN] Generating explanation for TRD-1003...

═══════════════════════════════════════════════════════════════════
  AGENTIC PnL ANOMALY DETECTION REPORT
═══════════════════════════════════════════════════════════════════
  [1] TRD-1003  |  2024-11-01  |  Equity desk
      PnL: $-712k
      Flags: extreme_loss, z_score_outlier (z=-4.21)

  EXPLANATION:
  This trade has been flagged because its daily loss of $712k is
  more than 4 standard deviations below the Equity desk's average,
  ...
```
