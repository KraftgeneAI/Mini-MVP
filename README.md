# KraftgeneAI | EnergyEminence (Mini-MVP)

### Core Demonstrator: Physics-Informed GNN for Power Grid Cascade Failure Prediction

**EnergyEminence (Mini-MVP)** is a streamlined AI dashboard demonstrating the core capability to predict cascading failures in power grids. Serving as the initial architectural validation, this tool utilizes a Physics-Informed Graph Neural Network (GNN) to process telemetry data (SCADA, PMU) and visualize critical failure propagation paths in real-time.

---

## 🚀 Key Features

### 1. **Real-Time Cascade Forecasting**
* **Physics-Informed Inference:** Combines GNNs with grid physics (frequency/voltage constraints) to predict failure propagation paths.
* **Sliding Window Analysis:** Simulates real-time monitoring by processing data in sequential time steps.
* **Dynamic Risk Scoring:** Evaluates 7 dimensions of risk (Threat, Vulnerability, Impact, etc.) for every node.

### 2. **Interactive Visualization (Operator Mode)**
* **Topology Rendering:** Drag, pan, and zoom through the grid structure.
* **Trajectory Mapping:** Visualizes the predicted "path" of the cascade with directional indicators.
* **Alert System:** Automated classification of system states (Nominal, Instability, Critical Cascade).

![EnergyEminence Dashboard - Cascade Propagation View](/images/screenshot2.JPG)

### 3. **Engineer / Debug Mode**
* **Ground Truth Overlay:** Visual comparison between Model Predictions (Red) and Historical Ground Truth (Green).
* **Deep Diagnostics:**
    * **Confusion Matrix:** Real-time Precision/Recall tracking.
    * **Signal Analysis:** Voltage histograms and frequency drift detection.
    * **Node-Level Probability Inspector:** Granular table view of raw model confidence scores vs. actual outcomes.

![EnergyEminence Dashboard - Engineer Diagnostics View](/images/screenshot1.JPG)

---

## 🎯 MVP Scope & Current Capabilities

This Mini-MVP release is scoped to demonstrate the feasibility of the AI architecture:
* **Topology:** Validated on the IEEE 118-bus test system (simulated environment).
* **Physics:** Integrates primary frequency and voltage constraints (simplified for real-time inference speed).
* **Data:** Pre-loaded with specific high-variance failure scenarios (e.g., `scenario_0.pkl`) to showcase detection logic.

---

## 🛠️ Installation & Setup

### 1. Prerequisite: Install Git LFS
This project uses large data files. Please ensure you have Git LFS installed before cloning.
- **Windows:** Download from [git-lfs.github.com](https://git-lfs.github.com/)
- **Mac:** `brew install git-lfs`
- **Linux:** `sudo apt-get install git-lfs`

Once installed, set it up once:

```bash
git lfs install
```
### 1. Clone the Repository
```bash
git clone https://github.com/KraftgeneAI/Mini-MVP
cd Mini-MVP
```

### 2. Create a Virtual Environment

It is best practice to run this project in an isolated environment.

**Linux / macOS:**

```bash
python -m venv venv
source venv/bin/activate
```

**Windows:**

```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies

Install the required PyTorch and scientific computing libraries.

```bash
pip install torch torch_geometric numpy matplotlib scipy tqdm psutil plotly streamlit networkx pandas
```

> **Note:** If you are using a GPU, ensure you install the CUDA-compatible version of PyTorch specific to your system. Visit [pytorch.org](https://pytorch.org/) for the exact command.

---

## 🚦 How to Run

### 1. Launch the Dashboard

Run the Streamlit application from your terminal:

```bash
streamlit run app.py
```

### 2. Using the Interface

1. **Select Data Source:** Use the sidebar to pick a scenario file (e.g., `scenario_0.pkl`) from the `data/` folder.
2. **Start Simulation:** Click **"▶ NEXT TIME STEP"** to advance the sliding window and generate predictions for cascade failures.
3. **Toggle Modes:**
* **Operator Mode (Default):** Clean view for situational awareness. Focuses on high-level alerts and predicted paths.
* **Engineer Mode:** Switch the toggle in the sidebar to reveal ground truth comparisons, confusion matrices, and physics diagnostics.

---

## 📂 Project Structure

```text
grid-guardian/
├── app.py                  # Main Streamlit Dashboard application
├── inference.py            # Core Inference Logic (Model Wrapper)
├── multimodal_cascade_model.py # GNN Architecture Definition
├── cascade_dataset.py      # Data Loader & Collate Functions
├── best_f1_model.pth       # Pre-trained Model Weights
├── grid_topology.pkl       # NetworkX Graph Structure of the Grid
├── data/                   # Folder containing .pkl scenario files
└── README.md               # Documentation
```
---

## 🔬 Technical Details

**The Model:**
The system uses a **Spatio-Temporal Graph Neural Network** that embeds grid topology (edges/nodes) and learns propagation patterns from historical cascade data. It integrates hard physical constraints (simplified/simulated) into the loss function to ensure realistic predictions.

**Metrics:**

* **Confidence:** The model's certainty that a cascade sequence is active.
* **Grid Frequency:** Monitored deviation from the nominal 60.0 Hz.
* **Voltage Stability:** Distribution of per-node voltage (p.u.) to detect collapse conditions.

---

## 📜 License & Status

**Project Status:** *Mini-MVP / Alpha Release.*
This codebase represents the initial architectural validation. Full enterprise integration, SCADA live-streaming adapters, and N-1 contingency analysis modules are currently in development for the v1.0 release.

© 2025 Kraftgene AI Inc. All Rights Reserved.
Proprietary.

---

**[View Project on GitHub](https://github.com/KraftgeneAI)**
