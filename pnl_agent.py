"""
=============================================================
  Agentic PnL Anomaly Detection System
  Built with LangGraph + RAG (ChromaDB) + Anthropic Claude
=============================================================

ARCHITECTURE — 4-node state machine:

  [ingest] → [detect] → [retrieve] → [explain]
       ↑                                  |
       └──────── (next batch) ────────────┘

Each node receives the full AgentState dict and returns an
updated copy. LangGraph wires the transitions for us.

HOW TO RUN:
  1. pip install langgraph langchain-anthropic chromadb
  2. Set ANTHROPIC_API_KEY in your environment
  3. python pnl_agent.py
"""

import os
import json
import statistics
from typing import TypedDict, Optional

# ── LangGraph ──────────────────────────────────────────────
from langgraph.graph import StateGraph, END

# ── LangChain / Claude ─────────────────────────────────────
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

# ── Vector store (RAG) ────────────────────────────────────
import chromadb
from chromadb.utils import embedding_functions

# ── Local data generator ──────────────────────────────────
import sys
sys.path.insert(0, os.path.dirname(__file__))
from data.generate_trades import generate_trades


# ══════════════════════════════════════════════════════════
#  STATE DEFINITION
#  TypedDict makes the schema explicit — LangGraph requires this.
#  Think of AgentState as a shared notebook that every node
#  can read and write to.
# ══════════════════════════════════════════════════════════

class AgentState(TypedDict):
    trades:           list[dict]          # raw trade records (input)
    desk_stats:       dict                # mean/stdev per desk (computed in detect)
    anomalies:        list[dict]          # flagged trades
    current_anomaly:  Optional[dict]      # the one being explained right now
    retrieved_context: str               # RAG output for current anomaly
    explanations:     list[dict]          # final outputs — one per anomaly
    anomaly_index:    int                 # pointer: which anomaly we're on


# ══════════════════════════════════════════════════════════
#  RAG SETUP
#  We build a tiny ChromaDB collection from the knowledge-base
#  text file. In production this would be a persistent store
#  with thousands of documents.
# ══════════════════════════════════════════════════════════

def build_rag_store(kb_path: str) -> chromadb.Collection:
    """
    Parse the knowledge-base text file into chunks and index
    them in an in-memory ChromaDB collection.
    """
    print("📚 Building RAG knowledge base...")

    client = chromadb.Client()                   # in-memory; use PersistentClient for disk

    # Use the default embedding function (sentence-transformers under the hood)
    ef = embedding_functions.DefaultEmbeddingFunction()

    collection = client.get_or_create_collection(
        name="pnl_knowledge",
        embedding_function=ef,
    )

    # Read and chunk by "---" delimiters
    with open(kb_path) as f:
        raw = f.read()

    chunks = [c.strip() for c in raw.split("---") if c.strip()]

    collection.add(
        documents=chunks,
        ids=[f"chunk_{i}" for i in range(len(chunks))],
    )

    print(f"  Indexed {len(chunks)} knowledge chunks.")
    return collection


# ══════════════════════════════════════════════════════════
#  NODE 1: INGEST
#  Loads trade data. In this demo it uses the synthetic generator.
#  In production, swap this for a database query.
# ══════════════════════════════════════════════════════════

def node_ingest(state: AgentState) -> AgentState:
    print("\n[NODE 1: INGEST] Loading trade data...")

    trades = generate_trades(n_days=10, n_trades_per_day=8)
    print(f"  Loaded {len(trades)} trades across {len(set(t['trade_date'] for t in trades))} trading days.")

    return {
        **state,
        "trades":          trades,
        "anomalies":       [],
        "explanations":    [],
        "anomaly_index":   0,
    }


# ══════════════════════════════════════════════════════════
#  NODE 2: DETECT
#  Runs two complementary anomaly detectors:
#    A) Z-score: flags trades > 2.5 std deviations from desk mean
#    B) Rule-based: hard rules (zero PnL, extreme absolute values)
#
#  Why both? Z-score catches statistical outliers. Rules catch
#  domain-specific patterns that stats might miss (e.g., exact zero).
# ══════════════════════════════════════════════════════════

def node_detect(state: AgentState) -> AgentState:
    print("\n[NODE 2: DETECT] Running anomaly detection...")

    trades = state["trades"]

    # ── Compute per-desk statistics ──────────────────────
    desk_pnls: dict[str, list[float]] = {}
    for t in trades:
        desk_pnls.setdefault(t["desk"], []).append(t["daily_pnl"])

    desk_stats = {}
    for desk, pnls in desk_pnls.items():
        if len(pnls) >= 2:
            desk_stats[desk] = {
                "mean":  statistics.mean(pnls),
                "stdev": statistics.stdev(pnls) or 1.0,   # avoid div-by-zero
            }

    # ── Flag anomalies ───────────────────────────────────
    anomalies = []
    Z_THRESHOLD       = 2.5     # flag if |z-score| > 2.5
    EXTREME_LOSS      = -300    # USD thousands
    EXTREME_GAIN      = 400     # USD thousands

    for trade in trades:
        desk  = trade["desk"]
        pnl   = trade["daily_pnl"]
        flags = []

        # Z-score check
        if desk in desk_stats:
            mean  = desk_stats[desk]["mean"]
            stdev = desk_stats[desk]["stdev"]
            z     = (pnl - mean) / stdev
            if abs(z) > Z_THRESHOLD:
                flags.append(f"z_score_outlier (z={z:.2f})")

        # Rule-based checks
        if pnl == 0.0:
            flags.append("zero_pnl_on_active_position")
        if pnl < EXTREME_LOSS:
            flags.append(f"extreme_loss (${pnl:,.0f}k)")
        if pnl > EXTREME_GAIN:
            flags.append(f"extreme_gain (${pnl:,.0f}k)")

        if flags:
            anomalies.append({**trade, "flags": flags})

    print(f"  Detected {len(anomalies)} anomalies from {len(trades)} trades.")
    for a in anomalies[:5]:
        print(f"  → {a['trade_id']} | {a['desk']} | PnL: ${a['daily_pnl']:,.0f}k | {a['flags']}")
    if len(anomalies) > 5:
        print(f"  ... and {len(anomalies) - 5} more.")

    return {
        **state,
        "desk_stats": desk_stats,
        "anomalies":  anomalies,
    }


# ══════════════════════════════════════════════════════════
#  NODE 3: RETRIEVE
#  For the current anomaly, query ChromaDB for the most
#  relevant knowledge chunks. This is the RAG step.
#
#  We build a natural-language query from the anomaly flags
#  so the semantic search finds the right section of the docs.
# ══════════════════════════════════════════════════════════

def node_retrieve(state: AgentState, collection: chromadb.Collection) -> AgentState:
    anomalies = state["anomalies"]
    idx       = state["anomaly_index"]

    if idx >= len(anomalies):
        # No more anomalies — signal completion
        return {**state, "current_anomaly": None, "retrieved_context": ""}

    anomaly = anomalies[idx]
    print(f"\n[NODE 3: RETRIEVE] RAG lookup for {anomaly['trade_id']} (flags: {anomaly['flags']})")

    # Build a query from the anomaly properties
    flag_text = " ".join(anomaly["flags"])
    pnl_desc  = "large loss" if anomaly["daily_pnl"] < 0 else "large gain" if anomaly["daily_pnl"] > 200 else "zero or unusual PnL"
    query     = f"{flag_text} {pnl_desc} {anomaly['instrument']} {anomaly['desk']} desk"

    results = collection.query(query_texts=[query], n_results=2)
    docs    = results["documents"][0]        # top-2 matching chunks

    context = "\n\n".join(docs)
    print(f"  Retrieved {len(docs)} knowledge chunks.")

    return {
        **state,
        "current_anomaly":   anomaly,
        "retrieved_context": context,
    }


# ══════════════════════════════════════════════════════════
#  NODE 4: EXPLAIN
#  Calls Claude with the anomaly details + RAG context.
#  The prompt is carefully structured so the LLM stays
#  grounded in the retrieved docs (reducing hallucination).
# ══════════════════════════════════════════════════════════

def node_explain(state: AgentState, llm: ChatAnthropic) -> AgentState:
    anomaly = state["current_anomaly"]
    context = state["retrieved_context"]

    if anomaly is None:
        return state

    print(f"\n[NODE 4: EXPLAIN] Generating explanation for {anomaly['trade_id']}...")

    system_prompt = """You are a senior risk analyst at an investment bank.
Your job is to explain PnL anomalies clearly to non-technical stakeholders.
Use ONLY the provided context documents to inform your explanation.
Be concise (3-4 sentences max). End with one concrete recommended action."""

    user_prompt = f"""ANOMALY DETECTED:
Trade ID:    {anomaly['trade_id']}
Date:        {anomaly['trade_date']}
Desk:        {anomaly['desk']}
Instrument:  {anomaly['instrument']}
Notional:    ${anomaly['notional_usd']:,.0f}
Daily PnL:   ${anomaly['daily_pnl']:,.0f}k
Flags:       {', '.join(anomaly['flags'])}

REFERENCE DOCUMENTS (use these to ground your explanation):
{context}

Write a plain-English explanation of why this trade was flagged and what to do next."""

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ])

    explanation = {
        "trade_id":    anomaly["trade_id"],
        "trade_date":  anomaly["trade_date"],
        "desk":        anomaly["desk"],
        "pnl":         anomaly["daily_pnl"],
        "flags":       anomaly["flags"],
        "explanation": response.content,
    }

    print(f"  ✓ Explanation generated ({len(response.content)} chars).")

    return {
        **state,
        "explanations":  state["explanations"] + [explanation],
        "anomaly_index": state["anomaly_index"] + 1,
    }


# ══════════════════════════════════════════════════════════
#  GRAPH WIRING
#  LangGraph's StateGraph connects nodes with directed edges.
#  The conditional edge after EXPLAIN loops back to RETRIEVE
#  until all anomalies have been processed.
# ══════════════════════════════════════════════════════════

def should_continue(state: AgentState) -> str:
    """
    Conditional edge: if there are more anomalies to explain,
    go back to retrieve. Otherwise, end.
    """
    if state["anomaly_index"] < len(state["anomalies"]):
        return "retrieve"
    return END


def build_graph(collection: chromadb.Collection, llm: ChatAnthropic) -> any:
    """
    Assemble the LangGraph state machine.

    Graph structure:
        ingest → detect → retrieve → explain
                              ↑          |
                              └──────────┘  (loops until all anomalies done)
    """
    graph = StateGraph(AgentState)

    # Add nodes — wrap retrieve/explain to inject dependencies
    graph.add_node("ingest",   node_ingest)
    graph.add_node("detect",   node_detect)
    graph.add_node("retrieve", lambda s: node_retrieve(s, collection))
    graph.add_node("explain",  lambda s: node_explain(s, llm))

    # Add edges (fixed transitions)
    graph.set_entry_point("ingest")
    graph.add_edge("ingest",  "detect")
    graph.add_edge("detect",  "retrieve")
    graph.add_edge("retrieve", "explain")

    # Conditional loop: explain → retrieve (more anomalies) or END
    graph.add_conditional_edges("explain", should_continue, {
        "retrieve": "retrieve",
        END:        END,
    })

    return graph.compile()


# ══════════════════════════════════════════════════════════
#  REPORT FORMATTER
# ══════════════════════════════════════════════════════════

def print_report(final_state: AgentState) -> None:
    explanations = final_state["explanations"]
    trades       = final_state["trades"]
    anomalies    = final_state["anomalies"]

    print("\n" + "═" * 65)
    print("  AGENTIC PnL ANOMALY DETECTION REPORT")
    print("═" * 65)
    print(f"  Trades analysed : {len(trades)}")
    print(f"  Anomalies found : {len(anomalies)}")
    print(f"  Explained       : {len(explanations)}")
    print("═" * 65)

    for i, exp in enumerate(explanations, 1):
        pnl_str = f"${exp['pnl']:+,.0f}k"
        print(f"\n{'─' * 65}")
        print(f"  [{i}] {exp['trade_id']}  |  {exp['trade_date']}  |  {exp['desk']} desk")
        print(f"      PnL: {pnl_str}")
        print(f"      Flags: {', '.join(exp['flags'])}")
        print(f"\n  EXPLANATION:")
        # Word-wrap at 60 chars
        words = exp["explanation"].split()
        line  = "  "
        for w in words:
            if len(line) + len(w) > 62:
                print(line)
                line = "  " + w + " "
            else:
                line += w + " "
        if line.strip():
            print(line)

    print(f"\n{'═' * 65}")
    print("  End of report.")
    print("═" * 65)

    # Save to JSON
    output_path = "pnl_anomaly_report.json"
    with open(output_path, "w") as f:
        json.dump({
            "summary": {
                "total_trades":   len(trades),
                "anomalies_found": len(anomalies),
                "explained":       len(explanations),
            },
            "anomalies": explanations,
        }, f, indent=2)
    print(f"\n  Full report saved → {output_path}")


# ══════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════

def main():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: Set ANTHROPIC_API_KEY environment variable first.")
        print("  export ANTHROPIC_API_KEY=sk-ant-...")
        return

    # 1. Build RAG store
    kb_path    = os.path.join(os.path.dirname(__file__), "knowledge_base", "pnl_concepts.txt")
    collection = build_rag_store(kb_path)

    # 2. Instantiate Claude
    llm = ChatAnthropic(
        model="claude-haiku-4-5-20251001",   # fast + cheap; swap for sonnet for higher quality
        max_tokens=400,
        api_key=api_key,
    )

    # 3. Build and run the LangGraph agent
    print("\n🚀 Starting PnL anomaly detection agent...")
    app = build_graph(collection, llm)

    initial_state: AgentState = {
        "trades":            [],
        "desk_stats":        {},
        "anomalies":         [],
        "current_anomaly":   None,
        "retrieved_context": "",
        "explanations":      [],
        "anomaly_index":     0,
    }

    final_state = app.invoke(initial_state)

    # 4. Print report
    print_report(final_state)


if __name__ == "__main__":
    main()
