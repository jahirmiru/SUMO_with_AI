# SUMO TIA Micro-Sim Orchestrator 🚦

An agentic, unified traffic microsimulation software designed to automate Traffic Impact Assessments (TIA). This software utilizes [Eclipse SUMO](https://eclipse.dev/sumo/) for physics-based traffic simulation, and integrates with **Ollama** to provide AI-accelerated traffic parameter optimizations, trip generation matrices, and signal configuration tuning.

## ✨ Features

- **Data Ingestion & Mapping:** Upload your CSV/Excel traffic volume data and assign intersection legs visually using an interactive map.
- **Trip Forecasting:** AI-distributed baseline and future trip matrix generation.
- **Automated Network Compiler:** Generates XML network models natively using SUMO's `netconvert` C++ hooks directly from Python.
- **AI Orchestration & Mitigation:** Connects to Local LLMs (via Ollama) to analyze lane layouts, capacities, and automatically inject optimized parameters.
- **Embedded Dashboard:** Harnesses Custom Tkinter and TraCI to deploy real-time simulation tracking dashboards.

## ⚙️ System Requirements

To run this platform, you need to have external environment tools installed as this software operates them as engine runtimes. 

1. **Python 3.10+** (Recommended)
2. **Eclipse SUMO:** 
   - [Download SUMO](https://eclipse.dev/sumo/intro.html). 
   - Make sure that the `SUMO_HOME` environment variable is fully configured and the SUMO bin folder is added to your local system's **Path**.
3. **Ollama:**
   - Install **[Ollama](https://ollama.com/)** to run the local assistant for parameter optimization. 
   - Pull the required model by running the following in your terminal:  
     `ollama pull llama3:latest` 

## 📦 Installation
We highly recommend using a Python Virtual Environment to keep dependencies clean.

Clone the repository and set up your environment:
```bash
git clone https://github.com/yourusername/SUMO_with_AI.git
cd SUMO_with_AI

# Create and activate a virtual environment (Windows)
python -m venv .venv
.venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

## 🚀 Usage Guide

After setting up the system dependencies and the python dependencies, launch the orchestrator. 

If `streamlit` is not recognized as a command in your terminal, use the Python module executor instead:

```bash
python -m streamlit run app.py
```

### Navigating the Pipeline

1. **Start a New Project:** From the main launch window, create your project label. 
2. **Ingest Mapping Coordinates:** Upload your intersection spreadsheet, and pin down the center & endpoint boundaries dynamically via OSM.
3. **Trip Generation:** Add percentage growth margins. You can let the AI generate a fractional distribution of expected traffic volumes.
4. **Compile Network Files:** Once step #1 and #2 are solved, hitting "Compile" spins up background processes connecting to Netconvert and builds `network.net.xml` nodes & edges automatically. 
5. **AI Orchestration:** Enter the chat module where an AI reviews your parameters against HCM methodologies for efficiency gains. 
6. **Live Dashboard:** Executes the compiled simulation directly connected to `sumo-gui` over local ports, rendering the visualization natively inside Tkinter dashboards. 

## 📝 Folder Formatting Notice
To keep things clean, dynamic outputs like compiled scenario `.xml` configurations, project states, and databases will securely build auto-created tree graphs in the local `projects/` directory when the application operates. 

---
_Built natively on Python for autonomous Traffic Engineering._
