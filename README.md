# Hybrid RAG

## 1. Setup Environment

```bash
python -m venv env
env\Scripts\activate
pip install -r requirements.txt
```

## 2. Add Required Files

### 📌 Required Files to place inside the project folder:

#### 1. manual.pdf  
Your machine manual.

#### 2. Llama-3.2-1B-Instruct-IQ3_M.gguf

Download the GGUF model from HuggingFace:

🔗 **Download Model:**  
https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/blob/main/Llama-3.2-1B-Instruct-IQ3_M.gguf

After downloading, place it inside the project folder.

## 3. Ingest PDF

```bash
python ingest_manual.py
```

## 4. Run Query (Terminal RAG Search)

```bash
python query_api.py -q "How to add detergent?"
```

## 5. Run Streamlit UI

```bash
streamlit run streamlit.py
```

This will open the UI at:

http://localhost:8501
