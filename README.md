# Hybrid RAG 

## 1. Setup Environment

``` bash
python -m venv env
env\Scripts\activate
pip install -r requirements.txt
```

## 2. Add Required Files

Place `manual.pdf` and your `.gguf` model in the project folder.

## 3. Ingest PDF

``` bash
python ingest_manual.py
```

## 4. Run Query

``` bash
python query_api.py -q "How to add detergent?"
```



## 5. Run Streamlit UI

``` bash
streamlit run streamlit.py
```
