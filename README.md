# 🏥 AI Nurse Copilot — Pre-Visit Intelligence Briefing System

**Apollo Homecare Case Study — Part 3: Working Prototype**

## What This Does

A home healthcare nurse is about to visit a patient. The AI Copilot pulls together everything relevant from 6 fragmented data sources and delivers a synthesized, actionable briefing — in seconds, not the 10-15 minutes it takes manually.

### The Prototype Demonstrates:

1. **Multi-source data synthesis** — Patient data scattered across CRM/EHR, nurse visit notes (multilingual), WhatsApp family messages, IoT vitals, physician instructions, and lab reports is unified into a single briefing.

2. **Cross-source signal detection** — Patterns that are invisible from any single source become visible when data is connected. Example: rising blood glucose (vitals) + skipped evening medication (nurse note) + family reporting appetite loss (WhatsApp) + physician's conditional escalation rule (physician note) → urgent physician call needed.

3. **RAG-based retrieval** — ChromaDB indexes patient data chunks with semantic embeddings. For each briefing request, the most relevant chunks are retrieved via semantic similarity search.

4. **Clinical guardrails** — The system never diagnoses. It flags signals with specific data points, cites sources, ranks by severity, and suggests actions.

5. **Structured output** — Every briefing follows a 5-section format optimized for mobile reading by nurses in transit.

6. **Feedback loop** — Nurses can rate briefings and flag inaccuracies, creating a data stream for system improvement.

## Mock Patients

| ID | Patient | Scenario | Key Cross-Source Signal |
|---|---|---|---|
| P001 | Ramesh Sharma (72M) | Diabetic elderly, declining vitals | Metformin non-adherence + rising glucose + family-reported appetite loss → physician escalation threshold breached |
| P002 | Lakshmi Venkatesh (65F) | Post-surgical knee replacement | Wound redness increasing + temperature rising + evening fever (family report) → possible surgical site infection in diabetic patient |
| P003 | Mohammed Ismail (68M) | COPD, medication adherence issues | Evening inhaler non-compliance + SpO2 declining (95→91%) + rescue inhaler use tripling + depression worsening → early exacerbation risk |

## Architecture

```
┌─────────────────────────────────────────────┐
│              DATA LAYER                      │
│  demographics.json │ visit_notes.txt         │
│  whatsapp_messages.txt │ vitals.csv          │
│  physician_notes.txt │ lab_reports.json      │
│                                              │
│  → Load → Chunk → Normalize → Tag           │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────┐
│          INTELLIGENCE LAYER                  │
│                                              │
│  ChromaDB ──→ Semantic Retrieval ──→ Context │
│  (Vector      (Top-15 chunks)      Assembly  │
│   Store)                                     │
│                    │                         │
│                    ▼                         │
│             Claude API                       │
│        (System prompt with                   │
│         clinical guardrails)                 │
│                    │                         │
│                    ▼                         │
│          Output Validation                   │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────┐
│           DECISION LAYER                     │
│                                              │
│  📋 Patient Snapshot                         │
│  🔄 What Changed Since Last Visit            │
│  🚨 Risk Flags (Critical/Moderate/Low)       │
│  ✅ Recommended Actions                      │
│  👥 Family Communication Context             │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────┐
│           FEEDBACK LOOP                      │
│  👍/👎 Rating + Text Comments                │
│  → Feeds into retrieval tuning,             │
│    prompt improvement, gap identification    │
└─────────────────────────────────────────────┘
```

## Setup

### Prerequisites
- Python 3.10+
- Anthropic API key (get one at [console.anthropic.com](https://console.anthropic.com))

### Install & Run

```bash
# Clone or copy the prototype folder
cd prototype

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

Then:
1. Open the URL shown in terminal (usually http://localhost:8501)
2. Enter your Anthropic API key in the sidebar
3. Select a patient
4. Click "Generate Pre-Visit Briefing"

## Tech Stack

| Component | Tool | Why |
|---|---|---|
| UI | Streamlit | Rapid prototyping, mobile-friendly, easy deployment |
| Vector Store | ChromaDB (in-memory) | Lightweight, no server setup, built-in embeddings |
| LLM | Claude Sonnet via Anthropic API | Strong multilingual capability, instruction-following, structured output |
| Data | Mock files (JSON, CSV, TXT) | Simulates production data sources realistically |

## Files

```
prototype/
├── app.py                  # Streamlit UI — main application
├── data_layer.py           # Data loading, chunking, ChromaDB indexing
├── intelligence_layer.py   # RAG retrieval + Claude API prompt pipeline
├── requirements.txt        # Python dependencies
├── README.md               # This file
└── data/
    ├── patient_001/        # Ramesh Sharma — Diabetic elderly
    │   ├── demographics.json
    │   ├── visit_notes.txt
    │   ├── whatsapp_messages.txt
    │   ├── vitals.csv
    │   ├── physician_notes.txt
    │   └── lab_reports.json
    ├── patient_002/        # Lakshmi Venkatesh — Post-surgical
    │   └── ... (same structure)
    └── patient_003/        # Mohammed Ismail — COPD
        └── ... (same structure)
```

## Prototype vs. Production

| Aspect | Prototype | Production |
|---|---|---|
| Data | Mock files | Live CRM, EHR, WhatsApp API, IoT gateways |
| Vector DB | ChromaDB in-memory | Pinecone/Weaviate (persistent, managed) |
| Retrieval | Semantic similarity | Hybrid: semantic + temporal + source-type weighting |
| LLM | Prompt engineering | Prompt engineering → fine-tuned model (6+ months) |
| Guardrails | Prompt-level | Multi-layer: prompt + output parser + clinical rules |
| Feedback | Thumbs up/down + comments | Explicit + implicit + outcome tracking |
| Deployment | Streamlit Cloud | Docker/K8s, API gateway, auth, offline caching |

## ⚠️ Disclaimer

This is a **prototype for demonstration purposes only**. All patient data is simulated. Not for clinical use. The system provides decision support and does not diagnose medical conditions.
