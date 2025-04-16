from langchain.schema import Document
import json

def load_dataset(path: str):
    try:
        with open(path, 'r') as f:
            data = json.load(f)

        documents = []
        for item in data:
            content = f"Question: {item['question']}\nAnswer: {item['answer']}"
            documents.append(Document(page_content=content))

        return documents

    except json.JSONDecodeError as e:
        print(f"Erreur de décodage JSON: {e}")
        return []
    
documents = load_dataset("data/glaive_dataset.json")
print(f"[DEBUG] Documents chargés : {len(documents)}")

