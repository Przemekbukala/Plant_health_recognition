# Plant_health_recognition
Przemysław Bukała, Krzysztof Kowalik, Jakub Ledwoń

## Running the App

```bash
ollama serve &
uv run streamlit run src/app.py
```
Open http://localhost:8501 in your browser.

### Test Wikipedia 
Uses hardcoded English Wikipedia article titles in `src/diseases.py` 
```bash
uv run python scripts/test_wikipedia_all.py
```

### Test  (Wikipedia + Ollama)
```bash
ollama serve &
uv run python scripts/test_pipeline.py
```
##  Quick Start (Linux / WSL)

This project is managed by **`uv`** 
The script below will automatically install the necessary system tools (Ollama), download the AI model, set up the Python environment.

### Setup

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Start Ollama server
ollama serve > /dev/null 2>&1 &

# Download model
ollama pull llama3

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install Python dependencies
uv sync

#  Run LLM setup script 
uv run python scripts/setup_llm.py

# Verify that Ollama works correctly
ollama run llama3

```

### Setup (PowerShell)

```powershell
# Install Ollama 
winget install Ollama.Ollama --accept-source-agreements --accept-package-agreements

# Add variable to PATH
$ollamaPath = "$env:LOCALAPPDATA\Programs\Ollama"
[Environment]::SetEnvironmentVariable(
 "Path",
 [Environment]::GetEnvironmentVariable("Path", "User") + ";" + $ollamaPath,
"User"
)

# Start Ollama server 
Start-Process ollama -ArgumentList "serve" -WindowStyle Hidden

# Download model ==
ollama pull llama3

# Install uv (Python package manager)
irm https://astral.sh/uv/install.ps1 | iex

# Refresh PATH 
$env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" +
            [System.Environment]::GetEnvironmentVariable("Path","User")


# Install dependencies
uv sync

# Run LLM setup script 
uv run python scripts/setup_llm.py

# Verify model works 
ollama run llama3
```

### Example training run

```
uv run python src/cnn_trainer.py --epochs 15
```

### Testing ready models

For models_comparator.py to work, dataset needs to be fixed first

```
chmod a+x scripts/dataset_fix.sh
```

Then comparison on test examples can be executed

```
uv run python src/models_comparator.py
```
