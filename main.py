"""Benthic Photo Annotator — standalone FastAPI service.

Serves the annotation UI at / and provides a REST API at /api/.
Upload reef photos → YOLO11 classifies 100-point grid → annotate in browser → export CSV.
"""

import base64
import io
import os
import threading
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image
from typing import Optional

# ---------------------------------------------------------------------------
# Label maps
# ---------------------------------------------------------------------------
# Labels are sourced from the YOLO models via export_onnx.py and taxonomy_tree.json
# T1 (Tier-1): 8 broad benthic functional groups
# T3 (Tier-3): 67 fine-grained taxonomic/functional categories
# See: https://huggingface.co/NMFS-OSI/yolo11m-cls-noaa-pacific-benthic-cover-t3

T3_DESCRIPTIONS: Dict[str, str] = {
    "ACAS": "Acanthastrea sp",
    "ACBR": "Acropora (branching)",
    "ACTA": "Acropora (tabular)",
    "ASPP": "Asparagopsis sp",
    "ASSP": "Astreopora sp",
    "ASTS": "Astrea spp",
    "BGMA": "Blue-green macroalga",
    "BRMA": "Brown macroalgae",
    "CAUL": "Caulerpa sp",
    "CCAH": "Crustose Coralline Algae (Healthy)",
    "CCAR": "Crustose Coralline Algae (Rubble)",
    "CMOR": "Corallimorph",
    "COSP": "Coscinaraea sp",
    "CYPS": "Cyphastrea sp",
    "DICO": "Dictyota spp.",
    "DICT": "Dictyosphaeria sp",
    "DISP": "Diploastrea sp",
    "ECHP": "Echinopora sp",
    "EMA": "Encrusting Macroalgae",
    "ENC": "Encrusting hard coral",
    "FASP": "Favites spp.",
    "FAVS": "Favites sp",
    "FINE": "Fine sediment",
    "FOL": "Foliose hard coral",
    "FREE": "Free-living hard coral",
    "FUSP": "Fungia spp.",
    "GASP": "Galaxea sp",
    "GOAL": "Goniopora/Alveopora sp",
    "GONS": "Goniastrea sp",
    "GRMA": "Green macroalgae",
    "HALI": "Halimeda spp.",
    "HCOE": "Heliopora sp",
    "HYSP": "Hydnophora sp",
    "ISSP": "Isopora sp",
    "LEPT": "Leptastrea spp.",
    "LOBO": "Lobophora spp.",
    "LOBS": "Lobophyllia spp.",
    "LPHY": "Leptoria sp",
    "MICR": "Microdictyon sp",
    "MISP": "Millepora sp",
    "MOBF": "Mobile fauna",
    "MOBR": "Montipora (branching)",
    "MOEN": "Montipora (encrusting)",
    "MOFO": "Montipora foliose",
    "OCTO": "Octocoral",
    "PADI": "Padina sp",
    "PAEN": "Pavona encrusting",
    "PAMA": "Pavona massive",
    "PESP": "Porites (encrusting)",
    "PHSP": "Phymastrea sp",
    "PLSP": "Platygyra spp.",
    "POBR": "Porites (branching)",
    "POCS": "Pocillopora spp.",
    "POEN": "Porites encrusting",
    "POFO": "Porites foliose",
    "POMA": "Porites (massive)",
    "PSSP": "Psammocora sp",
    "RDMA": "Red macroalgae",
    "SAND": "Sand",
    "SP": "Sponge",
    "STYS": "Stylophora sp",
    "TUN": "Tunicate",
    "TURFH": "Turf Algae (High)",
    "TURFR": "Turf Algae (Rubble)",
    "TURS": "Turbinaria sp",
    "UPMA": "Upright macroalga",
    "ZO": "Zoanthid",
}

_BENTHIC_NS = uuid.UUID("12345678-1234-5678-1234-567812345678")


def code_to_uuid(code: str) -> str:
    return str(uuid.uuid5(_BENTHIC_NS, f"benthic.{code}"))


def label_info(code: str) -> dict:
    return {"id": code_to_uuid(code), "name": T3_DESCRIPTIONS.get(code, code), "code": code}


# ---------------------------------------------------------------------------
# Model state
# ---------------------------------------------------------------------------

model_t3 = None
model_t1 = None
model_lock = threading.Lock()
models_ready = False

_HERE = Path(__file__).parent
T3_PATH = os.environ.get("MODEL_T3", str(_HERE / "models" / "yolo11m_cls_noaa-pacific-benthic-t3.pt"))
T1_PATH = os.environ.get("MODEL_T1", str(_HERE / "models" / "yolo11m_cls_noaa-pacific-benthic-t.pt"))

# ---------------------------------------------------------------------------
# In-memory store
# ---------------------------------------------------------------------------

image_store: Dict[str, dict] = {}
custom_labels: Dict[str, dict] = {}   # code -> label_info dict + is_custom flag

_CUSTOM_NS = uuid.UUID("87654321-4321-8765-4321-876543218765")


def custom_code_to_uuid(code: str) -> str:
    return str(uuid.uuid5(_CUSTOM_NS, f"custom.{code}"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def generate_grid_points(width: int, height: int, n_cols: int = 10, n_rows: int = 10, patch_size: int = 112) -> List[dict]:
    margin = patch_size // 2
    x_pos = [int(margin + c * (width - 2 * margin) / (n_cols - 1)) for c in range(n_cols)] if n_cols > 1 else [width // 2]
    y_pos = [int(margin + r * (height - 2 * margin) / (n_rows - 1)) for r in range(n_rows)] if n_rows > 1 else [height // 2]
    return [{"id": str(uuid.uuid4()), "row": row_px, "column": col_px, "annotations": []}
            for row_px in y_pos for col_px in x_pos]


def generate_stratified_random_points(width: int, height: int, cell_rows: int = 2, cell_cols: int = 5, patch_size: int = 112) -> List[dict]:
    import random
    margin = patch_size // 2
    usable_w = width  - 2 * margin
    usable_h = height - 2 * margin
    cell_w = usable_w / cell_cols
    cell_h = usable_h / cell_rows
    points = []
    for row in range(cell_rows):
        for col in range(cell_cols):
            x = margin + int(col * cell_w + random.random() * cell_w)
            y = margin + int(row * cell_h + random.random() * cell_h)
            x = max(margin, min(width  - margin, x))
            y = max(margin, min(height - margin, y))
            points.append({"id": str(uuid.uuid4()), "row": y, "column": x, "annotations": []})
    return points


def classify_point(img: Image.Image, row: int, col: int, patch_size: int, model) -> List[dict]:
    half = patch_size // 2
    patch = img.crop((max(0, col - half), max(0, row - half),
                      min(img.width, col + half), min(img.height, row + half)))
    patch = patch.resize((224, 224), Image.LANCZOS)

    with model_lock:
        results = model.predict(source=patch, imgsz=224, verbose=False)
        result = results[0]

    annotations = []
    for idx, conf in zip(result.probs.top5, result.probs.top5conf.tolist()):
        code = result.names[idx]
        info = label_info(code)
        annotations.append({
            "id": str(uuid.uuid4()),
            "benthic_attribute": info["id"],
            "ba_gr": info["id"],
            "ba_gr_label": info["name"],
            "code": code,
            "is_confirmed": False,
            "is_machine_created": True,
            "score": round(conf, 4),
        })
    return annotations


def point_stats(points: List[dict]):
    confirmed = sum(1 for p in points if p.get("annotations") and p["annotations"][0].get("is_confirmed"))
    unclassified = sum(1 for p in points if not p.get("annotations"))
    return confirmed, len(points) - confirmed - unclassified, unclassified


# ---------------------------------------------------------------------------
# App lifecycle
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model_t3, model_t1, models_ready
    try:
        from ultralytics import YOLO
        if Path(T3_PATH).exists():
            print(f"Loading T3 model: {T3_PATH}")
            model_t3 = YOLO(T3_PATH)
            print(f"T3 ready — {len(model_t3.names)} classes")
        if Path(T1_PATH).exists():
            print(f"Loading T1 model: {T1_PATH}")
            model_t1 = YOLO(T1_PATH)
            print(f"T1 ready — {len(model_t1.names)} classes")
        models_ready = model_t3 is not None or model_t1 is not None
    except Exception as exc:
        print(f"ERROR loading models: {exc}")
    yield


app = FastAPI(title="KIT-CAT — Keep it Tiny Coral Annotation Tool", lifespan=lifespan)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ---------------------------------------------------------------------------
# API routes  (all under /api)
# ---------------------------------------------------------------------------

@app.get("/api/health")
def health():
    return {"status": "ok", "models_ready": models_ready, "t3_loaded": model_t3 is not None, "t1_loaded": model_t1 is not None}


@app.get("/api/labels")
def get_labels():
    """Return all known label codes from loaded models plus custom labels."""
    standard = [label_info(code) for code in T3_DESCRIPTIONS]
    # Add any extra codes the T1 model knows that aren't already in T3_DESCRIPTIONS
    if model_t1:
        for idx, code in model_t1.names.items():
            if code not in T3_DESCRIPTIONS and code not in custom_labels:
                standard.append(label_info(code))
    if model_t3:
        for idx, code in model_t3.names.items():
            if code not in T3_DESCRIPTIONS and code not in custom_labels:
                standard.append(label_info(code))
    seen = set()
    deduped = []
    for lbl in standard:
        if lbl["code"] not in seen:
            seen.add(lbl["code"])
            deduped.append(lbl)
    custom = [{**v, "is_custom": True} for v in custom_labels.values()]
    return {"labels": deduped, "custom": custom}


@app.post("/api/labels", status_code=201)
async def create_label(request: Request):
    body = await request.json()
    code = str(body.get("code", "")).strip().upper()
    name = str(body.get("name", "")).strip()
    if not code or not name:
        raise HTTPException(400, "code and name are required.")
    if code in T3_DESCRIPTIONS:
        raise HTTPException(409, f"Code '{code}' already exists as a standard label.")
    if code in custom_labels:
        raise HTTPException(409, f"Custom label '{code}' already exists.")
    entry = {"id": custom_code_to_uuid(code), "code": code, "name": name, "is_custom": True}
    custom_labels[code] = entry
    return JSONResponse(content=entry, status_code=201)


@app.post("/api/images", status_code=201)
async def upload_image(
    image: UploadFile = File(...),
    model_name: str = Form("t1"),
    grid_method: str = Form("noaa"),
    grid_rows: int = Form(2),
    grid_cols: int = Form(5),
):
    # Validate model selection
    model_name = model_name.lower().strip()
    if model_name == "t3" and model_t3:
        model = model_t3
    elif model_name == "t1" and model_t1:
        model = model_t1
    elif model_t1:
        model = model_t1
        model_name = "t1"
    elif model_t3:
        model = model_t3
        model_name = "t3"
    else:
        raise HTTPException(503, "Models not loaded yet — retry in a few seconds.")

    grid_method = grid_method.lower().strip()
    if grid_method != "noaa":
        if not (2 <= grid_rows <= 50):
            raise HTTPException(400, "grid_rows must be between 2 and 50.")
        if not (2 <= grid_cols <= 50):
            raise HTTPException(400, "grid_cols must be between 2 and 50.")

    raw = await image.read()
    try:
        img = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        raise HTTPException(400, "Could not decode image.")

    width, height = img.size
    patch_size = max(60, min(width, height) // 14)

    if grid_method == "noaa":
        points = generate_stratified_random_points(width, height, cell_rows=2, cell_cols=5, patch_size=patch_size)
        grid_rows, grid_cols = 2, 5
    elif grid_method == "stratified":
        points = generate_stratified_random_points(width, height, cell_rows=grid_rows, cell_cols=grid_cols, patch_size=patch_size)
    else:
        points = generate_grid_points(width, height, n_cols=grid_cols, n_rows=grid_rows, patch_size=patch_size)

    print(f"[annotator] {image.filename} ({width}×{height}) model={model_name} grid={grid_rows}×{grid_cols} patch={patch_size} — classifying {len(points)} points …")
    for point in points:
        point["annotations"] = classify_point(img, point["row"], point["column"], patch_size, model)

    confirmed, unconfirmed, unclassified = point_stats(points)
    image_id = str(uuid.uuid4())

    # Thumbnail
    tw = 256
    th = max(1, int(height * tw / width))
    thumb_buf = io.BytesIO()
    img.resize((tw, th), Image.LANCZOS).save(thumb_buf, format="JPEG", quality=70)
    thumb_b64 = "data:image/jpeg;base64," + base64.b64encode(thumb_buf.getvalue()).decode()

    # Full image
    img_buf = io.BytesIO()
    img.save(img_buf, format="JPEG", quality=85)
    img_b64 = "data:image/jpeg;base64," + base64.b64encode(img_buf.getvalue()).decode()

    record = {
        "id": image_id,
        "name": image.filename or image_id,
        "original_image_width": width,
        "original_image_height": height,
        "patch_size": patch_size,
        "model_used": model_name,
        "grid_rows": grid_rows,
        "grid_cols": grid_cols,
        "num_confirmed": confirmed,
        "num_unconfirmed": unconfirmed,
        "num_unclassified": unclassified,
        "image": img_b64,
        "thumbnail": thumb_b64,
        "points": points,
        "created_on": _now_iso(),
        "updated_on": _now_iso(),
    }
    image_store[image_id] = record
    print(f"[annotator] done — {confirmed} confirmed / {unconfirmed} unconfirmed / {unclassified} unclassified")
    return JSONResponse(content=record, status_code=201)


@app.get("/api/images")
def list_images():
    """List images without base64 payload (lightweight for sidebar)."""
    return [
        {k: v for k, v in r.items() if k not in ("image",)}
        for r in image_store.values()
    ]


@app.get("/api/images/{image_id}")
def get_image(image_id: str):
    record = image_store.get(image_id)
    if not record:
        raise HTTPException(404, "Image not found.")
    return record


@app.patch("/api/images/{image_id}")
async def save_annotations(image_id: str, request: Request):
    record = image_store.get(image_id)
    if not record:
        raise HTTPException(404, "Image not found.")
    body = await request.json()
    if "points" in body:
        record["points"] = body["points"]
        confirmed, unconfirmed, unclassified = point_stats(body["points"])
        record["num_confirmed"] = confirmed
        record["num_unconfirmed"] = unconfirmed
        record["num_unclassified"] = unclassified
        record["updated_on"] = _now_iso()
    return {k: v for k, v in record.items() if k != "image"}


@app.delete("/api/images/{image_id}", status_code=204)
def delete_image(image_id: str):
    if image_id not in image_store:
        raise HTTPException(404, "Image not found.")
    del image_store[image_id]
    return Response(status_code=204)


@app.post("/api/images/{image_id}/reclassify")
async def reclassify_image(image_id: str, request: Request):
    record = image_store.get(image_id)
    if not record:
        raise HTTPException(404, "Image not found.")

    body = await request.json()
    model_name  = str(body.get("model_name",  "t1")).lower().strip()
    grid_method = str(body.get("grid_method", "noaa")).lower().strip()
    grid_rows   = int(body.get("grid_rows",   2))
    grid_cols   = int(body.get("grid_cols",   5))

    if model_name == "t3" and model_t3:
        model = model_t3
    elif model_name == "t1" and model_t1:
        model = model_t1
    elif model_t1:
        model = model_t1; model_name = "t1"
    elif model_t3:
        model = model_t3; model_name = "t3"
    else:
        raise HTTPException(503, "Models not loaded yet.")

    raw_b64 = record["image"].split(",", 1)[1]
    img = Image.open(io.BytesIO(base64.b64decode(raw_b64))).convert("RGB")
    width, height = img.size
    patch_size = record.get("patch_size", max(60, min(width, height) // 14))

    if grid_method == "noaa":
        points = generate_stratified_random_points(width, height, cell_rows=2, cell_cols=5, patch_size=patch_size)
        grid_rows, grid_cols = 2, 5
    elif grid_method == "stratified":
        grid_rows = max(2, min(50, grid_rows))
        grid_cols = max(2, min(50, grid_cols))
        points = generate_stratified_random_points(width, height, cell_rows=grid_rows, cell_cols=grid_cols, patch_size=patch_size)
    else:
        grid_rows = max(2, min(50, grid_rows))
        grid_cols = max(2, min(50, grid_cols))
        points = generate_grid_points(width, height, n_cols=grid_cols, n_rows=grid_rows, patch_size=patch_size)

    print(f"[reclassify] {record['name']} model={model_name} grid={grid_method} {grid_rows}x{grid_cols} — {len(points)} points …")
    for point in points:
        point["annotations"] = classify_point(img, point["row"], point["column"], patch_size, model)

    confirmed, unconfirmed, unclassified = point_stats(points)
    record.update({
        "points":          points,
        "model_used":      model_name,
        "grid_rows":       grid_rows,
        "grid_cols":       grid_cols,
        "num_confirmed":   confirmed,
        "num_unconfirmed": unconfirmed,
        "num_unclassified": unclassified,
        "updated_on":      _now_iso(),
    })
    return {k: v for k, v in record.items() if k != "image"}


@app.get("/api/images/{image_id}/export.csv")
def export_csv(image_id: str):
    record = image_store.get(image_id)
    if not record:
        raise HTTPException(404, "Image not found.")

    model_used = record.get("model_used", "t3")
    grid_rows = record.get("grid_rows", 10)
    grid_cols = record.get("grid_cols", 10)
    rows = ["image_name,point_id,row,column,confirmed,code,label,score,machine_created,is_custom_label,model_used,grid_rows,grid_cols"]
    for point in record["points"]:
        ann = point["annotations"][0] if point.get("annotations") else None
        is_custom = "yes" if ann and not ann.get("is_machine_created") and ann.get("code", "") in custom_labels else "no"
        rows.append(",".join([
            f'"{record["name"]}"',
            point["id"],
            str(point["row"]),
            str(point["column"]),
            "yes" if ann and ann.get("is_confirmed") else "no",
            ann.get("code", "") if ann else "",
            f'"{ann.get("ba_gr_label", "")}"' if ann else '""',
            str(ann.get("score", "")) if ann else "",
            "yes" if ann and ann.get("is_machine_created") else "no",
            is_custom,
            model_used,
            str(grid_rows),
            str(grid_cols),
        ]))

    filename = record["name"].rsplit(".", 1)[0] + "_annotations.csv"
    return Response(
        content="\n".join(rows),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# ---------------------------------------------------------------------------
# Static files — serve the SPA (must be last)
# ---------------------------------------------------------------------------

app.mount("/", StaticFiles(directory="static", html=True), name="static")
