# Manual Query Application

This project allows querying information from a PDF manual using vector embeddings and an LLM. It consists of two main scripts:

- `ingest_manual.py`: Ingests the PDF manual into a vector database (Chroma) by processing each PDF page.
- `query_api.py`: Queries the vector database with natural language questions, uses an LLM (Llama) to generate answers from relevant context.

---

## Setup Instructions

1. **Clone or download the repository** to your local machine.

2. **Create and activate a Python virtual environment** (recommended):

```bash
python -m venv env
# Windows
.\env\Scripts\activate
# macOS/Linux
source env/bin/activate
```

3. **Install required dependencies**:

```bash
pip install -r requirements.txt
```

4. **Ensure the following files exist in the project directory**:
- `manual.pdf` : The PDF file to ingest.
- `Llama-3.2-1B-Instruct-IQ3_M.gguf` : The Llama model file.
- `chroma_db/` : Directory will be created automatically on ingestion.

---

## Running the Application

### 1. Ingest the PDF Manual into the Vector Database

```bash
python ingest_manual.py
```

This will process the `manual.pdf` pages, embed them with a sentence transformer, and store them in Chroma vector db.

### 2. Query the Manual Using Natural Language

Use the query_api.py script with the `-q` option to ask a question:

```bash
python query_api.py -q "your question here"
```

Example:

```bash
python query_api.py -q "What is rinse aid?"
```

The script will return an answer derived from the context of the manual.

### 3. Run the Streamlit Web App

If you want a web-based interface, you can run the Streamlit app with:

```bash
streamlit run streamlit.py
```

-------
streamlit run streamlit.py

---

## Testing

Currently, critical-path testing has been performed on the CLI query functionality.

To test:

- **Ingest manual script**: Verify it processes `manual.pdf` without errors and creates the vector database.
- **Query script**: Run query_api.py with `-q` to ensure it returns answers correctly.

Further thorough testing including edge cases can be performed if needed.

---

## Notes

- Ensure you have sufficient resources (CPU, RAM) to load and run the Llama model.
- Directory paths and file names are configurable by modifying constants in the scripts.
- The system currently uses a local Chroma vector database for embeddings.

---

## Dependencies

Key packages:

- chromadb
- sentence-transformers
- llama-cpp-python
- PyMuPDF
- tqdm

Refer to `requirements.txt` for exact versions.

---

## License

Specify your project license here.

---

If you have any questions or issues, please reach out.
