"""
Intelligence Layer — Prompt engineering + RAG retrieval + Claude API generation.
Deterministic workflow (not agent-based) as specified in the system design.
"""

import anthropic
import chromadb
from data_layer import (
    load_all_patient_data,
    build_patient_collection,
    retrieve_relevant_chunks,
)

# ── System Prompt (Clinical Guardrails) ────────────────────────────────────

SYSTEM_PROMPT = """You are the AI Nurse Copilot for Apollo Homecare — a clinical decision-support system that generates pre-visit intelligence briefings for home healthcare nurses.

## YOUR ROLE
You synthesize patient data from multiple fragmented sources (EHR, nurse visit notes, WhatsApp family messages, IoT vitals, physician instructions, lab reports) into a single, actionable pre-visit briefing.

## CLINICAL GUARDRAILS — MANDATORY
1. You are a DECISION-SUPPORT tool, NOT a diagnostic system. NEVER diagnose conditions. Use language like "possible indicator of", "warrants assessment for", "pattern consistent with".
2. ALWAYS cite which data source a signal came from. Example: "Blood glucose has been above 200 for 3 consecutive days [Source: Vitals Data] which exceeds Dr. Mehta's escalation threshold [Source: Physician Notes]."
3. When flagging risks, ALWAYS provide the specific data points, not vague warnings. Bad: "Glucose is high." Good: "Fasting glucose: 235 mg/dL (March 28), 218 mg/dL (March 26), 195 mg/dL (March 24) — 3-reading upward trend, crossing the 200 mg/dL threshold set by Dr. Mehta."
4. NEVER fabricate data. If information is not present in the provided context, say "No data available" for that section.
5. Preserve multilingual content naturally. If a family member said something in Hindi/Urdu/Telugu, include the original along with a translation.
6. Prioritize ACTIONABLE insights. Every risk flag should end with a concrete suggested action.
7. For medication issues, always note both the medication name AND the clinical implication of non-adherence.

## OUTPUT FORMAT
Generate the briefing in EXACTLY this 5-section structure using markdown:

### 📋 Section 1: Patient Snapshot
- Name, age, gender, preferred form of address
- Primary diagnoses (with year diagnosed)
- Active medications (name, dosage, frequency)
- Allergies
- Care plan summary and visit frequency
- Supervising physician and last review date
- Primary family contact info

### 🔄 Section 2: Since Last Visit — What Changed
- Summary of ALL events since the last nurse visit
- New WhatsApp messages from family (with original language preserved)
- Vitals trend shifts (cite specific numbers)
- New physician instructions
- Any lab results received
- Explicitly flag what is NEW vs. what is unchanged

### 🚨 Section 3: Risk Flags & Alerts
Rank each flag by severity:
- 🔴 CRITICAL: Requires immediate action (call physician, change medication, emergency protocol)
- 🟡 MODERATE: Requires attention during this visit (assess, discuss, monitor closely)
- 🟢 LOW: Note for awareness (continue monitoring, no immediate action)

Each flag must include:
- The signal (what was detected)
- The data source(s) (where the signal came from)
- The clinical context (why it matters)
- Suggested action (what the nurse should do)

### ✅ Section 4: Recommended Actions for This Visit
Prioritized checklist derived from:
- Pending physician instructions
- Overdue assessments
- Risk flags from Section 3
- Care plan milestones
Number each action in priority order.

### 👥 Section 5: Patient & Family Communication Context
- Who is the primary family contact and their concerns
- Recent communication tone and topics
- Patient's preferred way of being addressed
- Any emotional or social context relevant to the visit (e.g., patient mood, family dynamics)
- Tips for maintaining relationship continuity

## IMPORTANT
- Be concise but thorough. Nurses read this on a mobile phone in transit.
- Use bullet points for scannability.
- Bold the most critical information.
- Today's date for context: assume the briefing is being generated for the NEXT scheduled visit.
"""

# ── Briefing Generation Workflow ───────────────────────────────────────────

def build_retrieval_query(patient_data: dict) -> str:
    """
    Construct a retrieval query from patient context.
    Focuses on recent events, active conditions, and pending actions.
    """
    demo = patient_data["demographics"]
    conditions = ", ".join(demo["primary_diagnoses"])
    meds = ", ".join([m["name"] for m in demo["current_medications"]])

    query = (
        f"Patient {demo['name']} with {conditions}. "
        f"Current medications: {meds}. "
        f"Recent vitals trends, medication adherence, family communications, "
        f"physician instructions, risk signals, wound status, "
        f"changes since last visit, pending actions."
    )
    return query


def assemble_context(retrieved_chunks: list[str], patient_data: dict) -> str:
    """
    Assemble retrieved chunks into a structured context block.
    Organized by source type for clarity.
    """
    # Always include demographics as structured data (not retrieved)
    import json
    demo_section = (
        "=== PATIENT DEMOGRAPHICS (Structured Data) ===\n"
        + json.dumps(patient_data["demographics"], indent=2, ensure_ascii=False)
    )

    # Retrieved chunks (already tagged with source type by data_layer)
    retrieved_section = "\n\n=== RETRIEVED CONTEXT (from RAG — ranked by relevance) ===\n"
    for i, chunk in enumerate(retrieved_chunks, 1):
        retrieved_section += f"\n--- Chunk {i} ---\n{chunk}\n"

    return demo_section + "\n\n" + retrieved_section


def generate_briefing(
    patient_id: str,
    api_key: str,
    chroma_client: chromadb.Client,
) -> str:
    """
    Main workflow — deterministic pipeline:
    1. Load patient data
    2. Build ChromaDB collection (index)
    3. Construct retrieval query
    4. Retrieve relevant chunks via RAG
    5. Assemble context
    6. Generate briefing via Claude API
    7. Return structured briefing
    """
    # Step 1: Load all patient data
    patient_data = load_all_patient_data(patient_id)

    # Step 2: Index into ChromaDB
    collection = build_patient_collection(patient_id, chroma_client)

    # Step 3: Construct retrieval query
    query = build_retrieval_query(patient_data)

    # Step 4: Retrieve relevant chunks
    retrieved_chunks = retrieve_relevant_chunks(collection, query, n_results=15)

    # Step 5: Assemble context
    context = assemble_context(retrieved_chunks, patient_data)

    # Step 6: Generate via Claude API
    client = anthropic.Anthropic(api_key=api_key)

    user_message = (
        f"Generate a Pre-Visit Intelligence Briefing for patient {patient_data['demographics']['name']} "
        f"(ID: {patient_id}).\n\n"
        f"The briefing is for the nurse's NEXT scheduled visit. "
        f"Synthesize ALL the data below — connect signals across sources to identify patterns "
        f"that wouldn't be visible from any single source alone.\n\n"
        f"PAY SPECIAL ATTENTION TO:\n"
        f"- Cross-source signals (e.g., vitals trend + family WhatsApp observation + medication non-adherence)\n"
        f"- Physician conditional instructions that may now be triggered by current data\n"
        f"- Gradual trends that cross clinical thresholds\n"
        f"- Family observations that have clinical relevance\n\n"
        f"--- BEGIN PATIENT DATA ---\n\n{context}\n\n--- END PATIENT DATA ---"
    )

    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )

    # Step 7: Extract and return
    briefing_text = message.content[0].text
    return briefing_text


def validate_briefing(briefing: str) -> dict:
    """
    Basic output validation — checks that all 5 sections are present.
    Production version would include hallucination detection and source verification.
    """
    required_sections = [
        "Section 1: Patient Snapshot",
        "Section 2: Since Last Visit",
        "Section 3: Risk Flags",
        "Section 4: Recommended Actions",
        "Section 5: Patient & Family Communication",
    ]

    validation = {"valid": True, "missing_sections": [], "warnings": []}

    for section in required_sections:
        if section.lower() not in briefing.lower():
            # Be flexible with exact naming
            section_keyword = section.split(":")[1].strip().split()[0].lower()
            if section_keyword not in briefing.lower():
                validation["missing_sections"].append(section)
                validation["valid"] = False

    if len(briefing) < 500:
        validation["warnings"].append("Briefing seems unusually short.")

    return validation
