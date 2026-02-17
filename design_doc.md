Here is the comprehensive, revised design document for the **PIE Workbench**. It incorporates all the details, architectural decisions, and workflow specifications you provided, maintaining the requested level of detail and redundancy to ensure no context is lost.

---

# Design Document: PIE Workbench (GUI for Parkinson’s Analysis)

## 1. Executive Summary

**Project Name:** PIE Workbench

**Goal:** Create a user-friendly, modular, and aesthetically pleasing Graphical User Interface (GUI) for the PIE (Parkinson’s Insight Engine) and PIE-clean ecosystems.

**Target Audience:** Clinical researchers and data scientists focusing on Parkinson’s Disease (PD) who require advanced analytics (Machine Learning & Statistics) without writing code.

**Inspiration:** WEKA (functionality) meets Modern Web Apps (aesthetics/usability).

---

## 2. System Architecture

To achieve the goal of a "beautiful" UI that is also "modular" and extensible, a **Hybrid Desktop Architecture** is recommended. This approach leverages the best of modern web rendering for the interface while retaining the raw power of Python for data processing.

### The Technology Stack

* **Frontend (UI):** Electron + React (TypeScript)
* *Rationale:* Web technologies allow for the most modern, responsive, and beautiful interfaces (charts, interactive tables, drag-and-drop capabilities).


* **Backend (Logic):** Python (FastAPI)
* *Rationale:* Directly imports your existing `PIE` and `PIE-clean` Python libraries without complex bridging. It acts as a local API server running as a background process.


* **Wrapper (Desktop App):** Electron
* *Rationale:* Packages the Python backend and React frontend into a single installable `.exe` or `.dmg` file. It manages the lifecycle of the Python process so the user just double-clicks an icon.


* **Communication:** HTTP/REST (localhost)
* *Rationale:* Communication occurs via standard HTTP requests between the Electron window and the local FastAPI server. This is robust for handling heavy data payloads.



### High-Level Diagram

```mermaid
graph LR
    User[User] <--> Electron[Electron Window (React UI)]
    Electron <--> API[Local API (FastAPI)]
    API <--> Libs[PIE / PIE-clean Libraries]
    Libs <--> Storage[Data Storage]

```

---

## 3. Directory Structure

The new repository `PIE-Workbench` will house the GUI code and reference the existing repositories as submodules. This allows the core logic to be updated independently of the GUI.

```text
PIE-Workbench/
├── package.json                 # Electron dependencies
├── src/                         # React Frontend Code
│   ├── components/              # Reusable UI (drag-and-drop zones, toggle switches)
│   ├── views/                   # Main Screens (DataIngest, MLEngine, StatsLab)
│   └── services/                # API client to talk to Python
│
├── backend/                     # The Python Server (The "Brain")
│   ├── main.py                  # FastAPI entry point
│   ├── api/                     # Endpoints
│   │   ├── project.py           # Save/Load project state
│   │   ├── data.py              # Routes for PIE-clean ops
│   │   └── analysis.py          # Routes for PIE ML ops
│   │
│   └── core/                    # The Adapter Layer (CRITICAL for modularity)
│       ├── abstract_loader.py   # Base class for data loading
│       ├── ppmi_loader.py       # Wrapper for PIE-clean
│       └── mjff_other_loader.py # Future placeholder
│
├── lib/                         # GIT SUBMODULES
│   ├── PIE/                     # Points to your existing PIE repo
│   └── PIE-clean/               # Points to your existing PIE-clean repo
│
├── resources/                   # Icons, static assets
└── README.md

```

---

## 4. Modular Design Strategy

To satisfy the requirement of supporting future diseases (e.g., Alzheimer's) and data sources (non-MJFF), the GUI will **not** hardcode "PPMI". Instead, it will use the **Adapter Pattern**.

### The "Data Source" Adapter

The GUI will interact with an abstract `DataSource` class.

* **Interface (`backend/core/abstract_loader.py`):**
* `detect_modalities(path)`: Returns list of available data types (e.g., "Genetics", "Motor Scores").
* `validate_schema(path)`: Returns simple Bool or Error List.
* `clean_and_load(options)`: Returns the processed Pandas DataFrame (or path to cached CSV).


* **Implementation (`backend/core/ppmi_loader.py`):**
* Imports `pie_clean`.
* Translates the GUI's generic requests into `pie_clean` specific function calls.


* **Future:** `AlzheimersAdapter` can be added easily by implementing the same interface.

### The "Analysis" Adapter

* **Interface:** `train_model()`, `get_feature_importance()`, `generate_report()`.
* **Implementation:** `PIEPipelineAdapter` (wraps existing `PIE` logic).

---

## 5. User Workflow & UI/UX

The software follows a linear workflow ("The Workbench" metaphor) where the user brings raw materials, selects tools, and produces a product.

### Phase 1: The Project Hub (The Landing Page)

* **Concept:** Similar to opening an IDE.
* **Actions:** "Create New Analysis" or "Open Existing Project."
* **Configuration:**
* **Project Name:** (e.g., "UPDRS Prediction Study").
* **Disease Context:** Dropdown selection [Parkinson's (PPMI) | Alzheimer's (Future)].
* **Data Path:** File picker to select the local `PPMI/` raw data folder.



### Phase 2: Data Ingestion (The PIE-clean Interface)

* **Goal:** Visualize and control the `PIE-clean` process.
* **UI Layout:** Split Screen.
* **Left Panel (Ingredients):** A checklist of modalities detected in the source folder (e.g., "Biospecimen", "Motor Scores").
* **Right Panel (Health Check):**
* **Missingness Heatmap:** When a user clicks "Medical History", show a generic React-Plotly heatmap visualizing data density.
* *Why:* Researchers need to see if the data is viable before running ML.




* **Action:** User clicks "Merge & Process".
* **Backend Call:** Triggers `pie_clean.pipeline.merge_data()`.
* **Feedback:** A specialized log window showing cleaning steps (e.g., "Dropping subject 102 due to missing baseline...").



### Phase 3: The Fork (Choose Your Path)

The UI splits into two main tabs: **Machine Learning (ML)** and **Statistics**.

#### Path A: The ML Engine (Visualizing PIE)

This tab essentially builds the `pipeline.py` arguments visually.

1. **Target Selection:**
* Dropdown of all available columns (e.g., `COHORT`, `UPDRS_Score`).
* *Smart Feature:* If categorical target selected → Suggest "Classification"; If continuous → Suggest "Regression".


2. **Feature Selection & Leakage Control:**
* **Leakage Control:** Drag-and-drop interface. User moves columns like `subject_characteristics_APPRDX` into a "Leakage/Exclude" bin.
* **Selection Method:** Dropdown (FDR, K-Best) and slider for parameters.


3. **Model Arena:**
* **"Auto-Pilot" Mode:** Uses default PyCaret settings.
* **"Expert" Mode:** Checkboxes for specific algorithms (RF, XGBoost, CatBoost) and hyperparameter tuning toggles.


4. **Execution:**
* Large "Run Analysis" button.
* Progress bar parsing stdout/callbacks from PIE.


5. **Results View:**
* Parse the results JSON/CSV produced by PIE.
* Render interactive **ROC Curves** and **Confusion Matrices** natively in React (no iframes).
* **Feature Importance:** Horizontal bar chart where clicking a bar shows the raw distribution of that feature.



#### Path B: Statistical Workbench (New Functionality)

For researchers who require p-values and traditional plots rather than predictive models.

1. **Variable Explorer:** Sidebar list of all columns, categorized by type (Numeric vs. Categorical).
2. **The "Canvas":**
* User drags a variable (e.g., `UPDRS_Part_III`) to the **Y-Axis** dropzone.
* User drags a variable (e.g., `Group` - Control/PD) to the **X-Axis** dropzone.


3. **Auto-Solver:**
* Backend detects *Continuous Y + Categorical X (2 groups)* → Runs **T-Test**.
* Backend detects *Continuous Y + Categorical X (>2 groups)* → Runs **ANOVA**.
* Backend detects *Continuous Y + Continuous X* → Runs **Pearson Correlation**.


4. **Survival Analysis Module:**
* Specific tab for "Time-to-Event".
* Dropdowns for `Time Variable`, `Event Variable`, and `Grouping Variable`.
* Output: Interactive **Kaplan-Meier curve**.



---

## 6. Implementation Roadmap

### Milestone 1: The "Hello World" Pipeline (Weeks 1-2)

* **Goal:** Get Python communicating with Electron.
* **Backend Task:** Create an endpoint `/api/load_preview` that accepts a file path, uses pandas to read the first 50 rows of a CSV, and returns JSON.
* **Frontend Task:** Render that JSON in a React Table component (using TanStack Table).

### Milestone 2: Integrating PIE-clean (Weeks 3-4)

* **Goal:** Connect the Data Ingestion UI to the existing `PIE-clean` library.
* **Task:** Import `pie_clean` into the FastAPI server.
* **Task:** Build the "Modality Selector" UI.
* **Challenge - Long-running processes:**
* *Solution:* When the user clicks "Process", backend starts a thread and returns a `task_id`. Frontend polls `/api/status/{task_id}` every 2 seconds to update the progress bar.



### Milestone 3: The ML Wrapper (Weeks 5-6)

* **Goal:** Expose `PIE` functionality via API.
* **Task:** Wrap `PIE/pie/pipeline.py` classes (`FeatureSelector`, `ModelTrainer`) into API endpoints.
* **Constraint:** Do not just call `subprocess.run('python pipeline.py')` as it is brittle. Instantiate the PIE classes directly inside the FastAPI routes.
```python
# Example Backend Logic
from lib.PIE.pie.classifier import ModelTrainer

@app.post("/train")
def train_model(config: ConfigModel):
    trainer = ModelTrainer(config)
    results = trainer.run()
    return results

```



### Milestone 4: The Stats Module (Weeks 7+)

* **Goal:** Implement the new statistical features.
* **Backend Task:** Implement `scipy.stats` and `lifelines` (for survival analysis) logic.
* **Frontend Task:** Use `Recharts` or `Victory` for React plotting.

### Milestone 5: Polish & Packaging

* **Goal:** Production readiness.
* **Task:** Add robust error handling (e.g., "PPMI Directory not found").
* **Task:** Package using `PyInstaller` (for backend) and `electron-builder` (for frontend).