# Local English Fluency Coach: Phi-3 Fine-Tuning & Ollama Deployment

This repository contains the training script and configuration files to fine-tune a Phi-3-Mini model to act as a private English fluency coach. The model learns to detect common non-native English speaker mistakes, provide correct phrasing, and explain grammar rules simply.

## 🎯 Project Overview
Non-native English speakers often make predictable mistakes due to native language syntax interference (e.g., mixing prepositions, incorrect tense usage). This pipeline:
1. Fine-tunes **Phi-3-Mini-Instruct** using Unsloth (PEFT/LoRA) on a dataset of common errors.
2. Formats dataset training text via custom instruction-response templates.
3. Exports the final checkpoint into a quantized `q4_k_m` GGUF file.
4. Hosts the specialized model locally and privately using Ollama.

## 🛠️ Project Structure
```text
├── json_extraction_dataset_500.json  # 500+ samples of ESL mistakes & explanations
├── fine_tuning.ipynb                 # Google Colab Unsloth notebook 
├── Modelfile                         # Ollama configuration blueprint
└── README.md
```

## 📋 Data Schema
The model is trained on structural text blocks mapping directly to your training loop's format:
```text
### Input: <User text containing non-native errors>
### Output: {"corrected": "<Text>", "explanation": "<Rule explanation>"} <|endoftext|>
```

## 🏃‍♂️ Execution Workflow

### Step 1: Execute Fine-Tuning
1. Upload `fine_tuning.ipynb` and `json_extraction_dataset_500.json` to Google Colab.
2. Select a **GPU Hardware Accelerator** (Notebook is configured for a T4 instance).
3. Run all cells. Cell 10 will compile your trained LoRA layers and output `gguf_model/unsloth.Q4_K_M.gguf`.
4. Allow your browser to complete the automated download managed by `files.download()`.

### Step 2: Set Up Your Modelfile
Move the downloaded `.gguf` file to your local project directory. Create a text file named `Modelfile` with the following configuration:

```dockerfile
# Load the local fine-tuned weights
FROM ./unsloth.Q4_K_M.gguf

# Synchronize template markers used during fine-tuning
TEMPLATE "### Input: {{ .Prompt }}\n### Output: "

# Set systemic parameters to guide response length and creativity
PARAMETER temperature 0.3
PARAMETER top_p 0.9
PARAMETER stop "###"
PARAMETER stop "<|endoftext|>"
```

### Step 3: Register and Query via Ollama
Open your terminal inside the project directory containing your files and run:

```bash
# Register the model into Ollama's local registry
ollama create fluency-coach -f ./Modelfile

# Start an interactive CLI chat session
ollama run fluency-coach
```

## 🧰 Dependencies
- **Core Frameworks:** Unsloth, TRL, PEFT, Accelerate, Bitsandbytes
- **Quantization Pipeline:** GGUF (`q4_k_m`)
- **Base Architecture:** Phi-3-Mini-4k-Instruct
