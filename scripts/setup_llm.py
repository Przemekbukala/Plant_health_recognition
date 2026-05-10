import subprocess
import sys

def main():
    print("Checking Ollama installation...")

    try:
        subprocess.run(["ollama", "--version"], check=True)
    except Exception:
        print("Error: Ollama is not installed or not available in PATH.")
        sys.exit(1)

    print("Pulling llama3 model...")

    try:
        subprocess.run(["ollama", "pull", "llama3"], check=True)
    except subprocess.CalledProcessError:
        print("Error while downloading llama3 model")
        sys.exit(1)

    print("\nSetup completed successfully.")
    print("You can now run:")
    print("ollama run llama3")


if __name__ == "__main__":
    main()