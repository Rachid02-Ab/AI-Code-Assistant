from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

def create_vectorstore(docs):
    if not docs:
        raise ValueError("La liste de documents est vide. Impossible de créer un vectorstore.")

    # Configuration des embeddings
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
