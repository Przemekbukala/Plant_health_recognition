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

# 2. Start Ollama server 
Start-Process ollama -ArgumentList "serve" -WindowStyle Hidden

# 3. Download model ==
ollama pull llama3

# 4. Install uv (Python package manager)
irm https://astral.sh/uv/install.ps1 | iex

# 5. Refresh PATH 
$env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" +
            [System.Environment]::GetEnvironmentVariable("Path","User")


# 6. Install dependencies
uv sync

# 7. Run LLM setup script 
uv run python scripts/setup_llm.py

# 8. Verify model works 
ollama run llama3
```
@TODO checked if it works on powershell (Windows)
