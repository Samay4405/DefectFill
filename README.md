# DefectFill: TensorFlow Two-Phase Industrial Anomaly Pipeline

This project provides a foundational, real-time anomaly detection architecture for MVTec AD using:

- Phase 1: Synthetic defect generation with a Keras-compatible Stable Diffusion backend + controlled latent perturbations.
- Phase 2: Few-shot anomaly detection with DINOv2 features (keras-hub) and a TensorFlow PatchCore implementation.

## Project Structure

- `configs/default.yaml`: Main pipeline configuration.
- `src/defectfill/phase1_synthesis.py`: Synthetic anomaly generation.
- `src/defectfill/feature_extractor.py`: DINOv2 dense patch feature extraction.
- `src/defectfill/patchcore.py`: Memory bank + nearest-neighbor anomaly scoring.
- `src/defectfill/heatmap.py`: Multi-threshold piecewise anomaly localization.
- `src/defectfill/optimize.py`: Latency benchmark + TFLite export.
- `src/defectfill/pipeline.py`: End-to-end orchestration.
- `src/defectfill/cli.py`: Terminal entry point.

## Setup

### First-Time Setup (For First-Time Cloning)

Follow these steps if you're setting up the project for the first time:

**Step 1: Clone and Navigate**
```powershell
cd C:\path\to\your\projects
git clone <repository-url>
cd DefectFill
```

**Step 2: Create Virtual Environment**
```powershell
python -m venv .venv
```

**Step 3: Activate Virtual Environment**
```powershell
# On Windows PowerShell
.\.venv\Scripts\Activate.ps1

# If you get execution policy error, run once:
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

**Step 4: Install Python Dependencies**
```powershell
python -m pip install -r requirements.txt
python -m pip install -e .
```

**Step 5: Set Up Dataset**
- Download MVTec AD from [MVTec website](https://www.mvtec.com/company/research/datasets/mvtec-ad)
- Place the dataset in `C:\path\to\dataset\` (any location)
- Update `configs/default.yaml` line 2 with your dataset path (use **forward slashes**):
  ```yaml
  root: "C:/path/to/your/mvtec_ad"  # Use forward slashes, not backslashes
  ```

**Step 6: Install Frontend Dependencies**
```powershell
cd frontend
npm install
cd ..
```

**Step 7: Build Memory Bank (One-Time)**

First, **activate the venv** (if not already active):
```powershell
.\.venv\Scripts\Activate.ps1
```

Then build the memory bank:
```powershell
python -m defectfill.cli --config configs/default.yaml build-memory
```

Or use the full venv Python path:
```powershell
.\.venv\Scripts\python.exe -m defectfill.cli --config configs/default.yaml build-memory
```

This generates the DINOv2 feature memory bank. You should see output like:
```
{"memory_bank_path": "./artifacts/memory_bank/bottle_memory.npy", "tflite_path": "./artifacts/tflite/patchcore_distance.tflite"}
```

Congratulations! Setup is complete. Proceed to "Running the Application" below.

---

### Running the Application

The application has two components that must run simultaneously:

**Terminal 1: Start Backend Server**
```powershell
# Activate venv first:
.\.venv\Scripts\Activate.ps1

# Then start the backend:
python -m uvicorn defectfill.backend.app:app --host 0.0.0.0 --port 8000
```

Expected output:
```
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

To verify the backend is running, open a new terminal and run:
```powershell
Invoke-RestMethod -Uri http://127.0.0.1:8000/health
```

Should return: `{"status": "ok", "target_fps": 12, ...}`

**Terminal 2: Start Frontend Server**
```powershell
# Navigate to frontend directory
cd frontend

# Start the dev server:
npm run dev
```

Expected output:
```
▲ Next.js 15.5.14 ... ready - started server on 0.0.0.0:3000
```

**Step 3: Open Dashboard**
- Open your browser and navigate to: **http://localhost:3000**
- You should see the live anomaly detection dashboard streaming real-time frames
- Frames come from MVTec test set; anomalies are highlighted with heatmap overlays

---

## Switching Between Object Categories

The system can detect defects on any MVTec AD category. By default, it's configured for **bottle** detection.

### **Available Categories**

`bottle` • `cable` • `capsule` • `carpet` • `grid` • `hazelnut` • `leather` • `metal_nut` • `pill` • `screw` • `tile` • `toothbrush` • `transistor` • `wood` • `zipper`

### **How to Switch Categories**

**Step 1: Update Config File**

Edit `configs/default.yaml` line 3:

```yaml
dataset:
  root: "C:/path/to/your/mvtec_ad"
  category: "bottle"   # ← Change this to your desired category (e.g., "leather", "cable", etc.)
  image_size: 384
```

**Step 2: Rebuild Memory Bank (IMPORTANT!)**

This step is **REQUIRED** after changing categories. The memory bank must be rebuilt using the new category's training data:

```powershell
# Make sure your venv is ACTIVATED first:
# .\.venv\Scripts\Activate.ps1

# Then run:
python -m defectfill.cli --config configs/default.yaml build-memory
```

Or use the full venv Python path directly:

```powershell
.\.venv\Scripts\python.exe -m defectfill.cli --config configs/default.yaml build-memory
```

This will:
- ✓ Load images from the NEW category's `train/good/` folder
- ✓ Extract DINOv2 features for that specific product type
- ✓ Build a new memory bank (e.g., `leather_memory.npy`)
- ✓ Export TFLite artifacts for the new category

Expected output:
```
{"memory_bank_path": "./artifacts/memory_bank/leather_memory.npy", "tflite_path": "./artifacts/tflite/patchcore_distance.tflite"}
```

**Step 3: Restart Backend**

Stop the running backend (press **CTRL+C** in that terminal) and restart it:

```powershell
.\.venv\Scripts\python.exe -m uvicorn defectfill.backend.app:app --host 0.0.0.0 --port 8000
```

The dashboard will automatically reconnect. You're now detecting defects on the new category!

### **Example: Switch from Bottle to Leather**

```powershell
# Activate venv first:
.\.venv\Scripts\Activate.ps1

# 1. Update default.yaml: change category from "bottle" to "leather"

# 2. Rebuild memory bank
python -m defectfill.cli --config configs/default.yaml build-memory

# 3. Restart backend (CTRL+C then restart)
python -m uvicorn defectfill.backend.app:app --host 0.0.0.0 --port 8000

# Dashboard now detects LEATHER defects!
```

---

## MVTec AD Layout

Expected folder layout:

```text
mvtec_ad/
  bottle/
    train/good/*.png
    test/good/*.png
    test/broken_large/*.png
    ...
```

Edit `configs/default.yaml` to point `dataset.root` to your local MVTec AD path.

## CLI Commands (Legacy / For Testing)

For one-off inference testing without the web dashboard:

```powershell
# First, activate your venv:
.\.venv\Scripts\Activate.ps1

# Then run any of these commands:
python -m defectfill.cli --config configs/default.yaml synthesize

python -m defectfill.cli --config configs/default.yaml infer --defect-type broken_large

python -m defectfill.cli --config configs/default.yaml build-memory
```

Or use the batch script (also requires venv activated):

```powershell
.\.venv\Scripts\Activate.ps1

./scripts/run_pipeline.ps1 -Config configs/default.yaml -DefectType broken_large
```

**Note:** For the live streaming dashboard (recommended), use the "Running the Application" section above instead.

## Latency Notes (<80 ms target)

This code includes:

- `@tf.function(jit_compile=True)` graph compilation on extractor/scoring paths.
- TFLite export for distance scorer in `optimize.py`.
- Batch-1 benchmarking and warning when `max_latency_ms` is exceeded.

For strict real-time targets, tune:

- `dataset.image_size` (e.g., 224 or 256)
- DINOv2 preset size (smaller backbone)
- `patchcore.coreset_ratio`
- CPU vs GPU execution and TensorRT/XLA options

## Important Compatibility Note

`keras-hub` preset naming and KerasCV diffusion APIs can vary by version. If a preset name is unavailable, update `DinoConfig.preset` in `feature_extractor.py` and re-run.
