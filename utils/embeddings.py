from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os
from pathlib import Path

def create_vectorstore(docs):
    if not docs:
        raise ValueError("La liste de documents est vide. Impossible de créer un vectorstore.")

    # Chemin local pour le cache du modèle
    model_cache_path = Path("./model_cache")
    model_cache_path.mkdir(exist_ok=True)
    
    # Configuration des embeddings avec gestion d'erreur
    try:
        embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': False},
            cache_folder=model_cache_path
        )
        
        # Création du vectorstore FAISS
        vectorstore = FAISS.from_documents(
            documents=docs,
            embedding=embedding_model
        )
        
        return vectorstore
        
    except Exception as e:
        # Si échec, essayer en mode local seulement
        try:
            print("Tentative en mode local seulement...")
            embedding_model = HuggingFaceEmbeddings(
                model_name=str(model_cache_path/"sentence-transformers_all-MiniLM-L6-v2"),
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': False},
                local_files_only=True
            )
            
            return FAISS.from_documents(documents=docs, embedding=embedding_model)
            
        except Exception as local_error:
            raise RuntimeError(
                f"Échec du chargement du modèle. "
                f"Veuillez télécharger manuellement le modèle avec:\n"
                f"```python\n"
                f"from sentence_transformers import SentenceTransformer\n"
                f"model = SentenceTransformer('all-MiniLM-L6-v2', cache_folder='./model_cache')\n"
                f"```\n"
                f"Erreur originale: {str(e)}\n"
                f"Erreur locale: {str(local_error)}"
            )