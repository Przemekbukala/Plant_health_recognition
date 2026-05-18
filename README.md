# Plant_health_recognition
Przemysław Bukała, Krzysztof Kowalik, Jakub Ledwoń

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
# 1. Install Ollama 
winget install Ollama.Ollama --accept-source-agreements --accept-package-agreements

# 2. Add variable to PATH
$ollamaPath = "$env:LOCALAPPDATA\Programs\Ollama"
[Environment]::SetEnvironmentVariable(
 "Path",
 [Environment]::GetEnvironmentVariable("Path", "User") + ";" + $ollamaPath,
"User"
)

# 3. Start Ollama server 
Start-Process ollama -ArgumentList "serve" -WindowStyle Hidden

# 4. Download model ==
ollama pull llama3

# 5. Install uv (Python package manager)
irm https://astral.sh/uv/install.ps1 | iex

# 6. Refresh PATH 
$env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" +
            [System.Environment]::GetEnvironmentVariable("Path","User")


# 7. Install dependencies
uv sync

# 8. Run LLM setup script 
uv run python scripts/setup_llm.py

# 9. Verify model works 
ollama run llama3
```

### Example training run

```
uv run python src/cnn_trainer.py --epochs 15
```

