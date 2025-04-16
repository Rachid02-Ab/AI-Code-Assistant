from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

def build_prompt():
    template = """
Vous êtes un assistant expert en programmation Python.
Répondez à la question de manière claire, précise et utile, en utilisant uniquement les informations du contexte ci-dessous.
Si le contexte ne contient pas suffisamment d'information pour répondre, dites que vous ne savez pas.

You are an expert assistant in Python programming.
Answer the question clearly, precisely, and helpfully, using only the information in the context below.
If the context is not sufficient, say you don't know.

---------------------
{context}
---------------------

Question / Query:
{question}

Réponse / Answer:
"""
    return PromptTemplate(input_variables=["context", "question"], template=template)

def build_rag_chain(llm, retriever):
    # Créer la chaîne RAG avec le modèle et le récupérateur de Chroma
    prompt = build_prompt()
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt}
    )
    return chain
