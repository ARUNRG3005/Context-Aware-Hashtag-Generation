"""
api/main.py — FastAPI wrapper for India Hashtag Generator
Endpoints:
  GET  /              → health + system status
  POST /predict       → generate hashtags with confidence + source
  GET  /labels        → all model labels
  GET  /kg/stats      → knowledge graph statistics
  GET  /kg/lookup     → debug entity lookup by name
"""

import sys
import os
import sqlite3

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ml", "inference"))

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import psutil
import time
from predictor import HashtagPredictor, KG_DB

# ── App ────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="India Hashtag Generator API",
    description=(
        "Context-aware hashtag generation for Indian news.\n\n"
        "Uses RoBERTa-base fine-tuned on 9,000+ India news articles "
        "combined with a dynamic Knowledge Graph of 5,086 Indian entities "
        "pulled from Wikidata.\n\n"
        "**Five prediction layers:**\n"
        "1. ML Model — domain classification\n"
        "2. Suppression — remove false positives\n"
        "3. Sensitive Map — crime/social/environment precision\n"
        "4. KG Lookup — entity-specific tags\n"
        "5. Relationship Inference — GTvsRR, IndVsAus, BJPvsCongress"
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model once at startup
predictor = HashtagPredictor()

# Create static directory if it doesn't exist
os.makedirs(os.path.join(os.path.dirname(__file__), "static"), exist_ok=True)
app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static")), name="static")


# ── Request / Response models ──────────────────────────────────────────────

class TextInput(BaseModel):
    text:      str
    threshold: float = 0.3
    top_k:     int   = 10

    class Config:
        json_schema_extra = {
            "example": {
                "text":      "GT defeated RR by 7 wickets in IPL 2026 Qualifier 2.",
                "threshold": 0.3,
                "top_k":     10,
            }
        }


class HashtagDetail(BaseModel):
    hashtag:    str
    confidence: float
    source:     str


class PredictResponse(BaseModel):
    text:              str
    hashtags_normal:   list[dict]
    hashtags_relation: list[dict]
    sources:           dict[str, str]
    model_used:        str
    kg_used:           bool
    total_found:       int
    latency_ms:        float
    hardware_metrics:  dict
    explainability:    dict[str, list[str]]
    audit_log:         list[str] = []


# ── Routes ─────────────────────────────────────────────────────────────────

@app.get("/", summary="Health check and system status")
def root():
    kg_available = os.path.exists(KG_DB)
    kg_entities  = 0
    if kg_available:
        try:
            conn = sqlite3.connect(KG_DB)
            cur  = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM entities")
            kg_entities = cur.fetchone()[0]
            conn.close()
        except Exception:
            pass

    return {
        "status":        "ok",
        "model":         "roberta-base",
        "model_labels":  len(predictor.labels),
        "kg_available":  kg_available,
        "kg_entities":   kg_entities,
        "version":       "1.0.0",
        "layers": [
            "1. ML Model",
            "2. Suppression",
            "3. Sensitive Map",
            "4. KG Lookup",
            "5. Relationship Inference",
        ],
    }


@app.post("/predict", response_model=PredictResponse, summary="Generate hashtags")
def predict(body: TextInput):
    """
    Generate hashtags for any Indian news text.

    The `source` field tells you which layer produced each tag:
    - **relationship** — auto-generated pair tag (GTvsRR, IndVsAus)
    - **model+kg** — confirmed by both ML model and Knowledge Graph
    - **sensitive_map** — crime/social/environment keyword matched
    - **model** — ML model prediction only
    - **kg** — Knowledge Graph entity found in text
    """
    if not body.text.strip():
        raise HTTPException(status_code=400, detail="text cannot be empty")

    if len(body.text) > 15000:
        raise HTTPException(status_code=400, detail="Text too long (max 15000 chars)")

    import psutil
    import torch
    process = psutil.Process()
    process.cpu_percent(interval=None) # Initialize CPU tracking
    ram_start = process.memory_info().rss
    
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        vram_start = torch.cuda.memory_allocated()

    start_time = time.time()
    
    results = predictor.predict(
        text=body.text,
        threshold=body.threshold,
        top_k=body.top_k,
    )
    
    latency = (time.time() - start_time) * 1000

    # Calculate exact delta for this single generation
    cpu_delta = process.cpu_percent(interval=None)
    ram_end = process.memory_info().rss
    ram_delta_mb = max(0, (ram_end - ram_start) / 1024 / 1024)
    
    vram_delta_mb = 0
    if torch.cuda.is_available():
        vram_peak = torch.cuda.max_memory_allocated()
        vram_delta_mb = (vram_peak - vram_start) / 1024 / 1024

    kg_used = any(r["source"] in ("kg", "model+kg", "relationship") for r in results["hashtags"])

    # Segregate Normal vs Relationship tags
    normal_tags = []
    relation_tags = []
    explain_dict = {}
    
    words = body.text.lower().split()

    for r in results["hashtags"]:
        tag = r["hashtag"]
        source = r["source"]
        confidence = r["confidence"]
        
        # Simple Explainability: Find intersecting words
        tag_lower = tag.lower().replace("#", "")
        matches = [w for w in words if len(w) > 3 and (w in tag_lower or tag_lower in w)]
        if not matches and "vs" in tag_lower:
            parts = tag_lower.split("vs")
            matches = [w for w in words if any(p in w for p in parts if len(p) > 2)]
            
        explain_dict[tag] = list(set(matches))
        
        if source == "relationship" or "vs" in tag.lower():
            relation_tags.append({"hashtag": tag, "confidence": confidence, "source": source})
        else:
            normal_tags.append({"hashtag": tag, "confidence": confidence, "source": source})

    return {
        "text":              body.text,
        "hashtags_normal":   normal_tags,
        "hashtags_relation": relation_tags,
        "sources":           results["sources"],
        "model_used":        "roberta-base",
        "kg_used":           kg_used,
        "total_found":       len(results["hashtags"]),
        "latency_ms":        latency,
        "hardware_metrics":  {"cpu_percent": cpu_delta, "ram_mb": round(ram_delta_mb, 2), "vram_mb": round(vram_delta_mb, 2)},
        "explainability":    explain_dict,
        "audit_log":         results.get("audit_log", [])
    }

class BatchTextInput(BaseModel):
    texts:     list[str]
    threshold: float = 0.3
    top_k:     int   = 10

@app.post("/predict/batch", summary="Generate hashtags for multiple texts")
def predict_batch(body: BatchTextInput):
    if len(body.texts) > 50:
        raise HTTPException(status_code=400, detail="max 50 texts per batch")
    
    responses = []
    for text in body.texts:
        if not text.strip():
            responses.append({"text": text, "hashtags": []})
            continue
        res = predictor.predict(text, threshold=body.threshold, top_k=body.top_k)
        responses.append({
            "text": text,
            "hashtags": [r["hashtag"] for r in res["hashtags"]],
            "sources": res["sources"]
        })
    return {"results": responses}

class ClassifyInput(BaseModel):
    text: str

@app.post("/classify", summary="Classify domain of text without generating hashtags")
def classify(body: ClassifyInput):
    if not body.text.strip():
        raise HTTPException(status_code=400, detail="text cannot be empty")
    
    # Run through Layer 1 only and get the highest scored domain
    encoding = predictor.tokenizer(
        body.text,
        max_length=128,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    import torch
    with torch.no_grad():
        logits = predictor.model(
            input_ids=encoding["input_ids"].to(predictor.model.device),
            attention_mask=encoding["attention_mask"].to(predictor.model.device),
        ).logits
        probs = torch.sigmoid(logits).cpu().numpy()[0]
        
    # Get top label
    best_idx = int(probs.argmax())
    best_label = predictor.labels[best_idx]
    
    return {
        "text": body.text,
        "dominant_domain_label": best_label,
        "confidence": float(probs[best_idx])
    }

# ── Hardware Monitoring & Dashboard Endpoints ─────────────────────────────

@app.get("/api/metrics/hardware", summary="Hardware Telemetry")
def get_hardware_metrics():
    proc = psutil.Process(os.getpid())
    ram_mb = proc.memory_info().rss / (1024 * 1024)
    cpu_percent = proc.cpu_percent(interval=0.1)
    
    battery = psutil.sensors_battery()
    battery_info = {"percent": battery.percent, "plugged": battery.power_plugged} if battery else None
    
    try:
        import torch
        vram_mb = torch.cuda.memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0
        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
    except Exception:
        vram_mb = 0
        gpu_name = "N/A"

    return {
        "ram_mb": ram_mb,
        "cpu_percent": cpu_percent,
        "battery": battery_info,
        "vram_mb": vram_mb,
        "gpu": gpu_name
    }

@app.get("/api/data/files", summary="List datasets in workspace")
def list_data_files():
    """Lists CSVs and DBs for the Data Transparency Hub."""
    base = os.path.dirname(os.path.dirname(__file__))
    data_dir = os.path.join(base, "data")
    
    files = []
    for root, dirs, filenames in os.walk(data_dir):
        for f in filenames:
            if f.endswith(('.csv', '.db', '.json')):
                full_path = os.path.join(root, f)
                rel_path = os.path.relpath(full_path, base)
                size_mb = os.path.getsize(full_path) / (1024*1024)
                files.append({"name": f, "path": rel_path, "size_mb": round(size_mb, 2)})
    return {"files": files}

@app.get("/api/scheduler", summary="Scheduler Status")
def get_scheduler_status():
    """Returns the status of the weekly scraper scheduler."""
    base = os.path.dirname(os.path.dirname(__file__))
    scheduler_log = os.path.join(base, "logs", "scheduler.log")
    
    status = "Stopped"
    last_run = "Never"
    
    if os.path.exists(scheduler_log):
        with open(scheduler_log, "r", encoding="utf-8") as f:
            lines = f.readlines()
            if lines:
                last_run = lines[-1].strip()
                status = "Running" if "Sleeping" in last_run or "Scraping" in last_run else "Stopped"
                
    return {
        "status": status,
        "last_log": last_run
    }


@app.get("/labels", summary="List all model labels")
def get_labels():
    """Returns all hashtag labels the ML model was trained on."""
    return {
        "count":  len(predictor.labels),
        "labels": sorted(predictor.labels),
    }


@app.get("/kg/stats", summary="Knowledge Graph statistics")
def kg_stats():
    """Returns entity counts, domain breakdown and recent fetch history."""
    if not os.path.exists(KG_DB):
        raise HTTPException(status_code=404, detail="Knowledge Graph database not found")

    try:
        conn = sqlite3.connect(KG_DB)
        conn.row_factory = sqlite3.Row
        cur  = conn.cursor()

        cur.execute("SELECT COUNT(*) as c FROM entities")
        total_entities = cur.fetchone()["c"]

        cur.execute("""
            SELECT domain, COUNT(*) as count
            FROM entities
            GROUP BY domain
            ORDER BY count DESC
        """)
        by_domain = {row["domain"]: row["count"] for row in cur.fetchall()}

        cur.execute("SELECT COUNT(*) as c FROM tags")
        total_tags = cur.fetchone()["c"]

        cur.execute("SELECT COUNT(*) as c FROM relationships")
        total_relationships = cur.fetchone()["c"]

        cur.execute("""
            SELECT category, fetch_time, entities_added, status
            FROM fetch_log
            ORDER BY fetch_time DESC
            LIMIT 10
        """)
        recent_fetches = [dict(row) for row in cur.fetchall()]

        conn.close()

        return {
            "total_entities":      total_entities,
            "total_tags":          total_tags,
            "total_relationships": total_relationships,
            "entities_by_domain":  by_domain,
            "recent_fetches":      recent_fetches,
            "db_path":             KG_DB,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"KG query failed: {str(e)}")


@app.get("/kg/lookup", summary="Look up a specific entity in the Knowledge Graph")
def kg_lookup_endpoint(name: str = Query(..., description="Entity name or abbreviation to look up. Example: GT, Virat Kohli, BJP")):
    """
    Debug endpoint — check if a name/abbreviation is in the KG.

    Examples:
    - `/kg/lookup?name=GT`
    - `/kg/lookup?name=Virat Kohli`
    - `/kg/lookup?name=BJP`
    """
    if not os.path.exists(KG_DB):
        raise HTTPException(status_code=404, detail="Knowledge Graph not found")

    try:
        conn = sqlite3.connect(KG_DB)
        conn.row_factory = sqlite3.Row
        cur  = conn.cursor()

        # Try exact match, then case-insensitive
        cur.execute("""
            SELECT e.wikidata_id, e.name, e.entity_type,
                   e.domain, e.sub_domain, e.last_updated
            FROM entities e
            WHERE e.name = ? OR LOWER(e.name) = LOWER(?)
            LIMIT 5
        """, (name, name))
        entities = [dict(row) for row in cur.fetchall()]

        results = []
        for entity in entities:
            cur.execute("""
                SELECT tag, weight FROM tags
                WHERE entity_id = ?
                ORDER BY weight DESC
            """, (entity["wikidata_id"],))
            tags = [row["tag"] for row in cur.fetchall()]

            cur.execute("""
                SELECT predicate, object_id FROM relationships
                WHERE subject_id = ?
                LIMIT 10
            """, (entity["wikidata_id"],))
            rels = [
                {"predicate": row["predicate"], "object": row["object_id"]}
                for row in cur.fetchall()
            ]

            results.append({
                "entity":        entity,
                "tags":          tags,
                "relationships": rels,
            })

        conn.close()

        return {
            "query":   name,
            "found":   len(results),
            "results": results,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)