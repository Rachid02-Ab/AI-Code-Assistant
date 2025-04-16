from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os

def create_vectorstore(docs):
    if not docs:
        raise ValueError("La liste de documents est vide.")
    
    try:
        # Option 1: Essayer avec le modèle léger par défaut
        embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        return FAISS.from_documents(docs, embedding_model)
    
    except Exception as e:
        # Option 2: Fallback plus robuste
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer('paraphrase-MiniLM-L3-v2')
            embeddings = model.encode([doc.page_content for doc in docs])
            
            return FAISS.from_embeddings(
                text_embeddings=zip([doc.page_content for doc in docs], embeddings),
                embedding=model
            )
        except Exception as alt_e:
            raise RuntimeError(
                f"Échec de création du vectorstore.\n"
                f"Veuillez vérifier:\n"
                f"1. Votre connexion Internet\n"
                f"2. Les versions des bibliothèques\n"
                f"3. La mémoire disponible\n\n"
                f"Erreur originale: {str(e)}"
            )