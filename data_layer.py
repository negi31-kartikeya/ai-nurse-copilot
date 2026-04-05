"""
Data Layer — Loads, chunks, and indexes patient data into ChromaDB.
Simulates the production data ingestion pipeline using mock files.
"""

import json
import os
import csv
import hashlib
import numpy as np
from datetime import datetime
import chromadb
from chromadb.api.types import EmbeddingFunction, Documents, Embeddings


DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

# ── Local Embedding Function (no downloads required) ──────────────────────

class LocalTfidfEmbedding(EmbeddingFunction):
    """
    A local TF-IDF + hash-based embedding function for ChromaDB.
    
    Production would use a proper sentence transformer model (all-MiniLM-L6-v2).
    This prototype uses TF-IDF vectorization with dimensionality hashing to
    generate fixed-size embeddings locally without downloading any models.
    
    This demonstrates the RAG architecture pattern while keeping the prototype
    fully self-contained.
    """
    
    EMBED_DIM = 384  # Match MiniLM dimensions
    
    def __call__(self, input: Documents) -> Embeddings:
        embeddings = []
        for text in input:
            embeddings.append(self._embed_text(text))
        return embeddings
    
    def _embed_text(self, text: str) -> list[float]:
        """Generate a deterministic embedding from text using feature hashing."""
        # Tokenize into words
        words = text.lower().split()
        
        # Feature hashing into fixed dimensions
        vec = np.zeros(self.EMBED_DIM, dtype=np.float32)
        for word in words:
            h = int(hashlib.md5(word.encode()).hexdigest(), 16)
            idx = h % self.EMBED_DIM
            sign = 1 if (h // self.EMBED_DIM) % 2 == 0 else -1
            vec[idx] += sign
        
        # Add bigram features for better semantic capture
        for i in range(len(words) - 1):
            bigram = f"{words[i]}_{words[i+1]}"
            h = int(hashlib.md5(bigram.encode()).hexdigest(), 16)
            idx = h % self.EMBED_DIM
            sign = 1 if (h // self.EMBED_DIM) % 2 == 0 else -1
            vec[idx] += sign * 0.5
        
        # L2 normalize
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        
        return vec.tolist()


# Singleton instance
_embedding_fn = LocalTfidfEmbedding()

# ── Loaders ────────────────────────────────────────────────────────────────

def load_demographics(patient_dir: str) -> dict:
    with open(os.path.join(patient_dir, "demographics.json"), "r", encoding="utf-8") as f:
        return json.load(f)


def load_visit_notes(patient_dir: str) -> list[dict]:
    """Parse visit notes into individual note chunks."""
    filepath = os.path.join(patient_dir, "visit_notes.txt")
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    notes = []
    raw_notes = content.split("--- Visit Note:")
    for raw in raw_notes:
        raw = raw.strip()
        if not raw:
            continue
        # Extract date from first line
        lines = raw.split("\n")
        date_str = lines[0].split("(")[0].strip().rstrip(" -—")
        notes.append({
            "source_type": "visit_note",
            "date": date_str,
            "content": raw,
        })
    return notes


def load_whatsapp_messages(patient_dir: str) -> list[dict]:
    """Parse WhatsApp messages into individual message chunks."""
    filepath = os.path.join(patient_dir, "whatsapp_messages.txt")
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    messages = []
    # Split by message timestamp pattern
    lines = content.split("\n")
    current_msg = []
    current_date = ""

    for line in lines:
        if line.startswith("[202"):
            # Save previous message if exists
            if current_msg:
                messages.append({
                    "source_type": "whatsapp",
                    "date": current_date,
                    "content": "\n".join(current_msg),
                })
            current_msg = [line]
            # Extract date
            try:
                current_date = line.split(",")[0].strip("[")
            except Exception:
                current_date = ""
        elif line.startswith("---"):
            # Group header — skip but save previous
            if current_msg:
                messages.append({
                    "source_type": "whatsapp",
                    "date": current_date,
                    "content": "\n".join(current_msg),
                })
                current_msg = []
        elif line.strip():
            current_msg.append(line)

    # Don't forget last message
    if current_msg:
        messages.append({
            "source_type": "whatsapp",
            "date": current_date,
            "content": "\n".join(current_msg),
        })

    return messages


def load_vitals(patient_dir: str) -> dict:
    """Load vitals CSV and return as a single summary chunk + raw data."""
    filepath = os.path.join(patient_dir, "vitals.csv")
    rows = []
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    # Build a readable summary of vitals trends
    if not rows:
        return {"source_type": "vitals", "date": "", "content": "No vitals data available.", "raw": []}

    latest = rows[-1]
    earliest = rows[0]

    # Build content string
    content_lines = [f"Vitals data from {earliest['date']} to {latest['date']} ({len(rows)} readings):\n"]
    content_lines.append("=== Latest Reading ===")
    for k, v in latest.items():
        if v and v.strip():
            content_lines.append(f"  {k}: {v}")

    # Add last 5 readings as a mini-table
    content_lines.append("\n=== Recent Readings (last 5) ===")
    for row in rows[-5:]:
        parts = [f"{k}={v}" for k, v in row.items() if v and v.strip()]
        content_lines.append("  " + ", ".join(parts))

    # Add full data for trend analysis
    content_lines.append("\n=== Full Time Series ===")
    for row in rows:
        parts = [f"{k}={v}" for k, v in row.items() if v and v.strip()]
        content_lines.append("  " + ", ".join(parts))

    return {
        "source_type": "vitals",
        "date": latest["date"],
        "content": "\n".join(content_lines),
        "raw": rows,
    }


def load_physician_notes(patient_dir: str) -> dict:
    filepath = os.path.join(patient_dir, "physician_notes.txt")
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    # Extract date
    date_str = ""
    for line in content.split("\n"):
        if line.startswith("Date:"):
            date_str = line.replace("Date:", "").strip()
            break

    return {
        "source_type": "physician_notes",
        "date": date_str,
        "content": content,
    }


def load_lab_reports(patient_dir: str) -> dict:
    filepath = os.path.join(patient_dir, "lab_reports.json")
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Format into readable text
    lines = []
    for report in data.get("reports", []):
        lines.append(f"Lab Report — {report['date']} ({report['type']})")
        lines.append(f"Lab: {report['lab']}")
        lines.append("Results:")
        for r in report["results"]:
            flag = f" [{r['flag']}]" if r.get("flag") else ""
            ref = f" (ref: {r['reference_range']})" if r.get("reference_range") else ""
            lines.append(f"  {r['test']}: {r['value']} {r.get('unit', '')}{ref}{flag}")
        if report.get("physician_comments"):
            lines.append(f"Physician Comments: {report['physician_comments']}")
        lines.append("")

    if data.get("next_labs_due"):
        lines.append(f"Next Labs Due: {data['next_labs_due']}")

    return {
        "source_type": "lab_reports",
        "date": data["reports"][0]["date"] if data.get("reports") else "",
        "content": "\n".join(lines),
    }


# ── Aggregator ─────────────────────────────────────────────────────────────

def load_all_patient_data(patient_id: str) -> dict:
    """Load all data for a patient. Returns structured dict."""
    patient_dir = os.path.join(DATA_DIR, patient_id)
    demographics = load_demographics(patient_dir)
    visit_notes = load_visit_notes(patient_dir)
    whatsapp = load_whatsapp_messages(patient_dir)
    vitals = load_vitals(patient_dir)
    physician = load_physician_notes(patient_dir)
    labs = load_lab_reports(patient_dir)

    return {
        "patient_id": patient_id,
        "demographics": demographics,
        "visit_notes": visit_notes,
        "whatsapp_messages": whatsapp,
        "vitals": vitals,
        "physician_notes": physician,
        "lab_reports": labs,
    }


def get_all_patient_ids() -> list[str]:
    """Return sorted list of patient folder names."""
    return sorted([
        d for d in os.listdir(DATA_DIR)
        if os.path.isdir(os.path.join(DATA_DIR, d)) and d.startswith("patient_")
    ])


def get_patient_summary(patient_id: str) -> str:
    """Quick one-liner for patient selector dropdown."""
    demo = load_demographics(os.path.join(DATA_DIR, patient_id))
    diagnoses = ", ".join([d.split("(")[0].strip() for d in demo["primary_diagnoses"][:2]])
    return f"{demo['name']} ({demo['age']}{demo['gender'][0]}) — {diagnoses}"


# ── ChromaDB Indexing ──────────────────────────────────────────────────────

def _date_to_epoch(date_str: str) -> int:
    """Convert date string to epoch for temporal sorting. Best-effort."""
    for fmt in ("%Y-%m-%d", "%B %d, %Y", "%Y-%m-%d %H:%M"):
        try:
            return int(datetime.strptime(date_str.strip(), fmt).timestamp())
        except (ValueError, AttributeError):
            continue
    return 0


def build_patient_collection(patient_id: str, chroma_client: chromadb.Client) -> chromadb.Collection:
    """
    Index all data chunks for a patient into a ChromaDB collection.
    Uses ChromaDB's built-in embedding (all-MiniLM-L6-v2).
    """
    collection_name = f"patient_{patient_id.replace('patient_', '')}"

    # Delete existing collection if present (for re-indexing)
    try:
        chroma_client.delete_collection(collection_name)
    except Exception:
        pass

    collection = chroma_client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
        embedding_function=_embedding_fn,
    )

    patient_data = load_all_patient_data(patient_id)
    documents = []
    metadatas = []
    ids = []
    idx = 0

    # Demographics (single chunk)
    demo = patient_data["demographics"]
    demo_text = json.dumps(demo, indent=2, ensure_ascii=False)
    documents.append(f"[PATIENT DEMOGRAPHICS]\n{demo_text}")
    metadatas.append({
        "source_type": "demographics",
        "timestamp": 0,
        "patient_id": patient_id,
    })
    ids.append(f"{patient_id}_demo_{idx}")
    idx += 1

    # Visit notes (one chunk per visit)
    for note in patient_data["visit_notes"]:
        documents.append(f"[NURSE VISIT NOTE — {note['date']}]\n{note['content']}")
        metadatas.append({
            "source_type": "visit_note",
            "timestamp": _date_to_epoch(note["date"]),
            "patient_id": patient_id,
        })
        ids.append(f"{patient_id}_visit_{idx}")
        idx += 1

    # WhatsApp messages (one chunk per message)
    for msg in patient_data["whatsapp_messages"]:
        documents.append(f"[WHATSAPP MESSAGE — {msg['date']}]\n{msg['content']}")
        metadatas.append({
            "source_type": "whatsapp",
            "timestamp": _date_to_epoch(msg["date"]),
            "patient_id": patient_id,
        })
        ids.append(f"{patient_id}_wa_{idx}")
        idx += 1

    # Vitals (single chunk with all data)
    vitals = patient_data["vitals"]
    documents.append(f"[VITALS DATA]\n{vitals['content']}")
    metadatas.append({
        "source_type": "vitals",
        "timestamp": _date_to_epoch(vitals["date"]),
        "patient_id": patient_id,
    })
    ids.append(f"{patient_id}_vitals_{idx}")
    idx += 1

    # Physician notes (single chunk)
    phys = patient_data["physician_notes"]
    documents.append(f"[PHYSICIAN NOTES — {phys['date']}]\n{phys['content']}")
    metadatas.append({
        "source_type": "physician_notes",
        "timestamp": _date_to_epoch(phys["date"]),
        "patient_id": patient_id,
    })
    ids.append(f"{patient_id}_phys_{idx}")
    idx += 1

    # Lab reports (single chunk)
    labs = patient_data["lab_reports"]
    documents.append(f"[LAB REPORTS — {labs['date']}]\n{labs['content']}")
    metadatas.append({
        "source_type": "lab_reports",
        "timestamp": _date_to_epoch(labs["date"]),
        "patient_id": patient_id,
    })
    ids.append(f"{patient_id}_labs_{idx}")
    idx += 1

    collection.add(documents=documents, metadatas=metadatas, ids=ids)
    return collection


def retrieve_relevant_chunks(
    collection: chromadb.Collection,
    query: str,
    n_results: int = 15,
) -> list[str]:
    """
    Hybrid retrieval: semantic similarity via ChromaDB.
    Returns top-N most relevant chunks as strings.
    
    In production, this would add temporal recency weighting and 
    source-type weighting. For the prototype, we rely on semantic 
    similarity and retrieve generously to let the LLM synthesize.
    """
    results = collection.query(
        query_texts=[query],
        n_results=min(n_results, collection.count()),
    )

    if not results or not results["documents"]:
        return []

    # Return documents sorted by relevance (ChromaDB returns in relevance order)
    return results["documents"][0]
