from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os

def create_vectorstore(docs):
    if not docs:
        raise ValueError("La liste de documents est vide. Impossible de créer un vectorstore.")

    # Configuration des embeddings
    try:
        embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': False}
        )
        
        # Création du vectorstore FAISS
        vectorstore = FAISS.from_documents(
            documents=docs,
            embedding=embedding_model
        )
        
        return vectorstore
        
    except Exception as e:
        # Solution alternative avec SentenceTransformer directement
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer('all-MiniLM-L6-v2')
            embeddings = model.encode([doc.page_content for doc in docs])
            
            # Création du FAISS avec les embeddings pré-calculés
            vectorstore = FAISS.from_embeddings(
                text_embeddings=list(zip([doc.page_content for doc in docs], embeddings)),
                embedding=model  # ou utiliser une fonction lambda si nécessaire
            )
            return vectorstore
            
        except Exception as alt_e:
            raise RuntimeError(
                f"Échec du chargement du modèle. "
                f"Erreur originale: {str(e)}\n"
                f"Erreur alternative: {str(alt_e)}"
            )