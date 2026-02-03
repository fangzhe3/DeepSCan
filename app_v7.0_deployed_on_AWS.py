# app.py
tokenizer = AutoTokenizer.from_pretrained(model_name)
base_model = AutoModel.from_pretrained(model_name)
Genesis_Quant_model = ESM2Regressor(base_model, head_type=head_type)
Genesis_Quant_model.to(device)
Genesis_Quant_model.eval()
app = FastAPI(title="Genesis Quant model prediction")


# -------------------------
# Helpers
# -------------------------
class PredictRequest(BaseModel):
    sequence: str


def clean_sequence(seq: str) -> str:
    seq = seq.replace(" ", "").replace("\n", "").upper()
    return seq


def validate_sequence(seq: str) -> None:
    allowed = set("ACDEFGHIKLMNPQRSTVWY")  # 20 canonical AAs
    bad = [aa for aa in seq if aa not in allowed]
    if bad:
        unique_bad = sorted(set(bad))
        raise HTTPException(
            status_code=400,
            detail=f"Invalid amino acids in sequence: {''.join(unique_bad)}. "
                   f"Allowed: {''.join(sorted(allowed))}",
        )
    if len(seq) == 0:
        raise HTTPException(status_code=400, detail="Sequence is empty.")
    if len(seq) > 1023:
        raise HTTPException(status_code=400, detail="Sequence too long (max 1023 aa).")

from typing import List, Dict, Any, Tuple

def parse_multi_sequence_text(text: str) -> List[Dict[str, str]]:
    """
    Accepts:
      - FASTA (one or many records)
      - Or multiple raw sequences separated by blank lines
      - Or a single raw sequence

    Returns: list of dicts: [{"id": "...", "sequence": "..."}]
    """
    raw = text.strip()
    if not raw:
        return []

    lines = [ln.rstrip("\r") for ln in raw.splitlines()]

    # FASTA mode if any header line starts with ">"
    if any(ln.strip().startswith(">") for ln in lines):
        records = []
        cur_id = None
        cur_seq_parts = []
        auto_idx = 1

        for ln in lines:
            ln = ln.strip()
            if not ln:
                continue
            if ln.startswith(">"):
                # flush previous
                if cur_id is not None:
                    records.append({"id": cur_id, "sequence": "".join(cur_seq_parts)})
                header = ln[1:].strip()
                cur_id = header if header else f"seq_{auto_idx}"
                auto_idx += 1
                cur_seq_parts = []
            else:
                # allow spaces inside seq lines; remove them
                cur_seq_parts.append(ln.replace(" ", ""))

        # flush last
        if cur_id is not None:
            records.append({"id": cur_id, "sequence": "".join(cur_seq_parts)})

        return records

    # Non-FASTA: split by blank lines into records
    blocks = []
    buf = []
    for ln in lines:
        if ln.strip() == "":
            if buf:
                blocks.append("".join(buf))
                buf = []
        else:
            buf.append(ln.strip())
    if buf:
        blocks.append("".join(buf))

    if not blocks:
        return []

    out = []
    for i, b in enumerate(blocks, start=1):
        out.append({"id": f"seq_{i}", "sequence": b})
    return out


def longest_consecutive_signal_loose(labels, signal=1, gap_labels={0, 2}, max_gap=1):
    max_len = 0
    best_start = None
    best_end = None

    start = 0
    gap_count = 0

    for end, v in enumerate(labels):
        if v in gap_labels:
            gap_count += 1

        while gap_count > max_gap:
            if labels[start] in gap_labels:
                gap_count -= 1
            start += 1

        curr_len = end - start + 1
        if curr_len > max_len:
            max_len = curr_len
            best_start = start
            best_end = end

    if max_len == 0:
        return None, None, 0

    return best_start, best_end, max_len

def clean_and_validate_records(records: List[Dict[str, str]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Returns (valid_records, error_records)
    valid_records entries include: id, sequence, length
    error_records entries include: id, error
    """
    valid = []
    errors = []

    for r in records:
        rid = (r.get("id") or "").strip() or "seq"
        seq = clean_sequence(r.get("sequence") or "")

        try:
            validate_sequence(seq)
            valid.append({"id": rid, "sequence": seq, "length": len(seq)})
        except HTTPException as e:
            errors.append({"id": rid, "error": e.detail})

    return valid, errors


def append_results_to_daily_csv(results, model_version="Genesis_Quant_v7"):
    """
    Append prediction results to a per-day CSV file (UTC).
    Creates folder + header if they do not exist.
    """

    now = datetime.now(timezone.utc)
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.isoformat()

    # Folder per day
    base_dir = Path("results") / date_str
    base_dir.mkdir(parents=True, exist_ok=True)

    csv_path = base_dir / f"Genesis_Quant_predictions_{date_str}.csv"
    file_exists = csv_path.exists()

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        # Write metadata + header ONCE
        if not file_exists:
            writer.writerow([f"# model={model_version}"])
            writer.writerow([f"# date={date_str} (UTC)"])
            writer.writerow([
                "id",
                "sequence",
                "sequence_length",
                "Genesis_score",
                "Omni_score",
                "CST_category",
                "Topology",
                "TMD_seq",
                "request_time_utc",
            ])

        # Append rows
        for r in results:
            writer.writerow([
                r["id"],
                r["sequence"],
                int(r["sequence_length"]),
                float(r["Genesis_score"]),
                float(r["Omni_score"]),
                r["CST_category"],
                int(r["Topology"]),
                r["TMD_seq"],
                time_str,
            ])

import shutil
from datetime import datetime, timedelta, timezone, date

RESULTS_ROOT = Path("results")
RETENTION_DAYS = 7

def cleanup_old_results():
    """
    Delete result folders older than RETENTION_DAYS (UTC).
    Folder names must be YYYY-MM-DD.
    """
    if not RESULTS_ROOT.exists():
        return

    cutoff_date = datetime.now(timezone.utc).date() - timedelta(days=RETENTION_DAYS)

    for d in RESULTS_ROOT.iterdir():
        if not d.is_dir():
            continue

        try:
            folder_date = datetime.strptime(d.name, "%Y-%m-%d").date()
        except ValueError:
            # Skip non-date folders
            continue

        if folder_date < cutoff_date:
            shutil.rmtree(d, ignore_errors=True)

_last_cleanup_date = None

def cleanup_once_per_day():
    global _last_cleanup_date
    today = date.today()

    if _last_cleanup_date != today:
        cleanup_old_results()
        _last_cleanup_date = today

# -------------------------
# Routes
# -------------------------
@app.get("/", response_class=HTMLResponse)
async def index():
    # Single-page UI
    html = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="UTF-8" />
      <title>DeepSCan Cell Surface Display Sequence Predictor</title>
      <meta name="viewport" content="width=device-width, initial-scale=1.0" />
      <style>
        * { box-sizing: border-box; }
        body {
          margin: 0;
          font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
          background: #0f172a;
          color: #e5e7eb;
          display: flex;
          justify-content: center;
          align-items: flex-start;
          min-height: 100vh;
          padding: 24px;
        }
        .container {
          width: 100%;
          max-width: 960px;
          background: rgba(15,23,42,0.9);
          border-radius: 18px;
          border: 1px solid #1f2937;
          box-shadow: 0 20px 60px rgba(0,0,0,0.45);
          padding: 24px 28px 28px;
        }
        .header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 16px;
          gap: 12px;
        }
        .title {
          font-size: 1.4rem;
          font-weight: 650;
        }
        .subtitle {
          font-size: 0.9rem;
          color: #9ca3af;
        }
        .badge {
          font-size: 0.75rem;
          padding: 4px 8px;
          border-radius: 999px;
          border: 1px solid #4b5563;
          color: #d1d5db;
          background: rgba(31,41,55,0.85);
        }
        label {
          font-size: 0.9rem;
          display: block;
          margin-bottom: 6px;
          color: #d1d5db;
        }
        textarea {
          width: 100%;
          min-height: 160px;
          resize: vertical;
          border-radius: 12px;
          border: 1px solid #374151;
          background: #020617;
          padding: 10px 12px;
          color: #e5e7eb;
          font-family: "JetBrains Mono", "Consolas", monospace;
          font-size: 0.9rem;
          outline: none;
        }
        textarea:focus {
          border-color: #38bdf8;
          box-shadow: 0 0 0 1px rgba(56,189,248,0.5);
        }
        .help-text {
          margin-top: 4px;
          font-size: 0.8rem;
          color: #9ca3af;
        }
        .controls {
          margin-top: 12px;
          display: flex;
          align-items: center;
          gap: 12px;
          flex-wrap: wrap;
        }
        button {
          border: none;
          border-radius: 999px;
          padding: 8px 16px;
          font-weight: 600;
          font-size: 0.9rem;
          cursor: pointer;
          display: inline-flex;
          align-items: center;
          gap: 8px;
          background: linear-gradient(135deg, #22c55e, #0ea5e9);
          color: #0b1120;
        }
        button:disabled {
          opacity: 0.6;
          cursor: default;
        }
        .spinner {
          width: 14px;
          height: 14px;
          border-radius: 999px;
          border: 2px solid rgba(15,23,42,0.9);
          border-top-color: rgba(15,23,42,0.2);
          animation: spin 0.8s linear infinite;
        }
        @keyframes spin {
          to { transform: rotate(360deg); }
        }
        .status {
          font-size: 0.85rem;
          color: #9ca3af;
        }
        .error {
          margin-top: 10px;
          padding: 8px 10px;
          border-radius: 10px;
          background: rgba(127,29,29,0.25);
          border: 1px solid #b91c1c;
          color: #fecaca;
          font-size: 0.85rem;
        }
        .results {
          margin-top: 18px;
          padding: 14px 16px;
          border-radius: 14px;
          background: rgba(15,23,42,0.8);
          border: 1px solid #1f2937;
        }
        .results-header {
          display: flex;
          justify-content: space-between;
          align-items: baseline;
          margin-bottom: 8px;
        }
        .results-title {
          font-size: 1rem;
          font-weight: 600;
        }
        .results-meta {
          font-size: 0.75rem;
          color: #9ca3af;
        }
        .prob-row {
          display: flex;
          justify-content: space-between;
          align-items: center;
          font-size: 0.85rem;
          margin: 4px 0;
        }
        .prob-bar-wrapper {
          flex: 1;
          margin: 0 10px;
          height: 6px;
          border-radius: 999px;
          background: #020617;
          overflow: hidden;
        }
        .prob-bar {
          height: 100%;
          border-radius: inherit;
          background: linear-gradient(90deg, #0ea5e9, #22c55e);
          width: 0%;
        }
        .footer {
          margin-top: 16px;
          font-size: 0.75rem;
          color: #6b7280;
          text-align: right;
        }
        code {
          font-family: "JetBrains Mono", "Consolas", monospace;
          font-size: 0.8rem;
        }
      </style>
    </head>
    <body>
      <div class="container">
        <div class="header">
          <div>
            <div class="title">DeepSCan Cell Surface Display (CSD) Sequence Predictor</div>
            <div class="subtitle">Paste a protein sequence and get model scores in one click.</div>
          </div>
          <div class="badge">Genesis Quant · Omni models</div>
        </div>

        <div>
          <label for="sequence">Protein sequence (single-letter amino acids)</label>
          <textarea id="sequence" placeholder="Paste FASTA (multi) or raw sequences.
Example FASTA:
>seq1
MVHLTPEEKSAVTALWGKVNVDEVGGEALGRLL...
>seq2
MSTNPKPQRKTKRNTNRRPQDVKFPGGGQIVGGV..."></textarea>
          <div class="help-text">Max length per sequence: 1023 aa.</div>
        </div>

        <div class="controls">
          <button id="predictBtn">
            <div class="spinner" id="spinner" style="display:none;"></div>
            <span id="btnLabel">Run prediction</span>
          </button>
          
          <button id="downloadBtn"
            disabled
            style="background:linear-gradient(135deg,#a855f7,#6366f1);">
            Download CSV
          </button>

          <div class="status" id="statusText">Model ready.</div>
        </div>

        <div id="errorBox" class="error" style="display:none;"></div>

        <div id="results" class="results" style="display:none;">
          <div class="results-header">
            <div class="results-title">Prediction</div>
            <div class="results-meta" id="resultsMeta"></div>
          </div>
          <div id="scoreBox" style="font-size: 2rem; font-weight: 700; margin-top: 6px;"></div>
          <div id="scoreNote" style="font-size: 0.85rem; color:#9ca3af; margin-top: 6px;"></div>
        </div>

        <div class="footer">
          Copyright: Sidi Chen lab, Yale University
        </div>
      </div>

      <script>
        const predictBtn = document.getElementById("predictBtn");
        const spinner = document.getElementById("spinner");
        const btnLabel = document.getElementById("btnLabel");
        const statusText = document.getElementById("statusText");
        const errorBox = document.getElementById("errorBox");
        const results = document.getElementById("results");
        const resultsMeta = document.getElementById("resultsMeta");
        const sequenceInput = document.getElementById("sequence");
        const downloadBtn = document.getElementById("downloadBtn");
        
        let lastPredictionPayload = null;

        function setLoading(isLoading) {
          predictBtn.disabled = isLoading;
          spinner.style.display = isLoading ? "inline-block" : "none";
          btnLabel.textContent = isLoading ? "Predicting..." : "Run prediction";
          statusText.textContent = isLoading ? "Running model on server..." : "Model ready.";
        }

        function showError(msg) {
          errorBox.textContent = msg;
          errorBox.style.display = "block";
        }

        function clearError() {
          errorBox.style.display = "none";
          errorBox.textContent = "";
        }

        function renderResults(payload) {
            // payload: { n_total, n_scored, results: [...], errors: [...] }
            const results = document.getElementById("results");
            const resultsMeta = document.getElementById("resultsMeta");
            const scoreBox = document.getElementById("scoreBox");
            const scoreNote = document.getElementById("scoreNote");

            const { n_total, n_scored, results: rows, errors } = payload;

            resultsMeta.textContent = `Input: ${n_total} | Scored: ${n_scored} | Errors: ${errors.length}`;

            let html = `
              <div style="overflow:auto; margin-top:10px;">
                <table style="width:100%; border-collapse: collapse; font-size:0.9rem;">
                  <thead>
                    <tr style="text-align:left; border-bottom:1px solid #1f2937;">
                      <th style="padding:8px 6px;">ID</th>
                      <th style="padding:8px 6px;">Length</th>
                      <th style="padding:8px 6px;">Genesis_Score</th>
                      <th style="padding:8px 6px;">Omni_Score</th>
                      <th style="padding:8px 6px;">Genesis_CST_category</th>
                      <th style="padding:8px 6px;">Topology</th>
                    </tr>
                  </thead>
                  <tbody>
            `;

            for (const r of rows) {
              html += `
                <tr style="border-bottom:1px solid #111827;">
                  <td style="padding:8px 6px; white-space:nowrap;">${escapeHtml(r.id)}</td>
                  <td style="padding:8px 6px;">${r.sequence_length}</td>
                  <td style="padding:8px 6px; font-weight:650;">${Number(r.Genesis_score).toFixed(4)}</td>
                  <td style="padding:8px 6px; font-weight:650;">${Number(r.Omni_score).toFixed(4)}</td>
                  <td style="padding:8px 6px;">${r.CST_category}</td>
                  <td style="padding:8px 6px;">${r.Topology}</td>
                </tr>
              `;
            }

            html += `</tbody></table></div>`;

            if (errors.length) {
              html += `<div style="margin-top:12px; font-size:0.85rem; color:#fecaca;">
                <div style="font-weight:650; margin-bottom:6px;">Errors</div>
              `;
              for (const e of errors) {
                html += `<div><code>${escapeHtml(e.id)}</code>: ${escapeHtml(e.error)}</div>`;
              }
              html += `</div>`;
            }

            scoreBox.innerHTML = html;
            scoreNote.textContent = "Potent CSD sequence score is >0.8. Topology 1 is Type I, Extracellular domain at N term";
            results.style.display = "block";
        }

        function escapeHtml(str) {
            return String(str)
              .replaceAll("&", "&amp;")
              .replaceAll("<", "&lt;")
              .replaceAll(">", "&gt;")
              .replaceAll('"', "&quot;")
              .replaceAll("'", "&#039;");
        }

        predictBtn.addEventListener("click", async () => {
          clearError();
          results.style.display = "none";

          const seqRaw = sequenceInput.value.trim();
          if (!seqRaw) {
            showError("Please paste a protein sequence first.");
            return;
          }

          setLoading(true);
          try {
            const resp = await fetch("/predict", {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({ sequence: seqRaw })
            });
            
            
            const data = await resp.json();
            if (!resp.ok) {
              showError(data.detail || "Prediction failed.");
            } else {
              renderResults(data);
              lastPredictionPayload = data;   
              downloadBtn.disabled = false;
            }
          } catch (err) {
            showError("Network or server error: " + err);
          } finally {
            setLoading(false);
          }
        });
        
        downloadBtn.addEventListener("click", async () => {
          const resp = await fetch("/download_csv", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(lastPredictionPayload)
          });

          const blob = await resp.blob();
          const url = URL.createObjectURL(blob);
          
          const ts = new Date().toISOString().replace(/[:.]/g, "-");
          const a = document.createElement("a");
          a.download = `Genesis_Quant_predictions_${ts}.csv`;
          a.href = url;
          a.click();

          URL.revokeObjectURL(url);
        });
        
      </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html)


@app.post("/predict")
async def predict(req: PredictRequest):
    # req.sequence contains the raw textarea input (FASTA or not)
    raw_text = (req.sequence or "").strip()
    records = parse_multi_sequence_text(raw_text)
    if not records:
        raise HTTPException(status_code=400, detail="No sequences found. Paste FASTA or raw sequences.")

    valid, errors = clean_and_validate_records(records)
    if not valid:
        # all failed validation
        return {
            "n_total": len(records),
            "n_scored": 0,
            "results": [],
            "errors": errors,
        }

    seqs = [r["sequence"] for r in valid]

    try:
        tokens = tokenizer(
            seqs,                  # batch
            return_tensors="pt",
            padding=True,          # pad to max in batch
            truncation=False,      # we already enforce <=1023; don't silently cut
        )
        tokens = {k: v.to(device) for k, v in tokens.items()}
        
        with torch.no_grad():
            outputs = Genesis_Quant_model(**tokens)
            outputs2 = Omni_linear_model(**tokens)
            logits = outputs["logits"].detach().float().cpu().numpy().reshape(-1)
            logits2 = outputs2["quant_preds"].detach().float().cpu().numpy().reshape(-1)
            topo_logits = outputs2["topo_logits"].detach().float().cpu().numpy()
            tm_logits = outputs2["tm_logits"].detach().float().cpu().numpy()

        topo_preds = topo_logits.argmax(axis=1).astype(int)
        tm_preds = [tm.argmax(axis=1).astype(int) for tm in tm_logits]
        tmd_seqs = [extract_longest_segment_loose(seq, tm_pred) for seq, tm_pred in zip(seqs, tm_preds)]
            
        results = []
        bins = [-10, 0.3, 0.55, 0.8, 1, 1.2, 20]
        CSTs = pd.cut(logits, bins=bins, labels=["CST0", "CST1", "CST2", "CST3", "CST4", "CST5"], include_lowest=True,).astype(str)
        for r, score, score2, CST, topo, tmd_seq in zip(valid, logits, logits2, CSTs, topo_preds, tmd_seqs):
            results.append({
                "id": r["id"],
                "sequence": r["sequence"],
                "sequence_length": r["length"],
                "Genesis_score": float(score),
                "Omni_score": float(score2),
                "CST_category": CST,
                "Topology": int(topo),
                "TMD_seq": tmd_seq,
            })
        
        append_results_to_daily_csv(results)
        cleanup_once_per_day()
        return {
            "n_total": len(records),
            "n_scored": len(results),
            "results": results,
            "errors": errors,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model error: {e}")


@app.post("/download_csv")
async def download_csv(payload: dict):
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([f"# model=Genesis_Quant_v7"])
    writer.writerow([f"# timestamp={datetime.utcnow().isoformat()}"])
    writer.writerow(["id", "sequence", "sequence_length", "Genesis_score", "Omni_score", "CST_category", "Topology", "TMD_seq"])

    for r in payload["results"]:
        writer.writerow([r["id"], r["sequence"], r["sequence_length"], r["Genesis_score"], r["Omni_score"], r["CST_category"], r["Topology"], r["TMD_seq"]])

    output.seek(0)
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    filename = f"Genesis_Quant_predictions_{ts}.csv"
    return StreamingResponse(
        output,
        media_type="text/csv",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"'
        }
)
