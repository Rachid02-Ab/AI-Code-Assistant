import streamlit as st
import sys  
from pathlib import Path  

sys.path.append(str(Path(__file__).parent.parent))  
from utils.loader import load_dataset
from utils.splitter import split_documents
from utils.embeddings import create_vectorstore
from models.mistral_inference import get_mistral_response
from app.rag_pipeline import build_rag_chain
import re


# Titre de l'application Streamlit
st.title("AI Coder - Copilot pour Code Python")

# Introduction améliorée
st.markdown("""
Bienvenue dans le chatbot **AI Coder** ! 
Posez vos questions sur le code Python et obtenez des réponses basées sur un modèle **RAG**.

*Exemples de questions à essayer :*
- Comment créer une fonction en Python ?
- Comment installer Python sur AWS EC2 ?
- Comment formater du temps en heures:minutes:secondes ?
""")

# Chargement des données
with st.spinner("Chargement des connaissances..."):
    documents = load_dataset("data/glaive_dataset.json")
    docs = split_documents(documents)
    vectorstore = create_vectorstore(docs)
    retriever = vectorstore.as_retriever(search_type="similarity", k=3)

# Fonction pour formater les blocs de code
def format_response(response):
    # Détecter les blocs de code avec ```python ou ```
    response = re.sub(r'```python(.*?)```', r'<pre><code class="language-python">\1</code></pre>', response, flags=re.DOTALL)
    response = re.sub(r'```(.*?)```', r'<pre><code>\1</code></pre>', response, flags=re.DOTALL)
    return response

# Fonction RAG améliorée
def rag_pipeline(query):
    retrieved_docs = retriever.get_relevant_documents(query)
    context = "\n".join([doc.page_content for doc in retrieved_docs])
    
    prompt = f"""
Vous êtes un assistant expert en programmation Python.
Répondez en formatant proprement les blocs de code avec ```python avant et ``` après.

Contexte:
{context}

Question: {query}

Réponse (en Markdown, avec blocs de code si nécessaire):
"""
    
    response = get_mistral_response(prompt)
    return response

# Interface utilisateur
user_question = st.text_area("Entrez votre question sur le code Python:", "", height=100)

if st.button("Soumettre"):
    if user_question.strip():
        with st.spinner("Recherche et génération de la réponse..."):
            try:
                response = rag_pipeline(user_question)
                
                st.markdown("### 💡 Réponse")
                
                # Afficher les parties normales en markdown
                st.markdown(format_response(response), unsafe_allow_html=True)
                
                # Afficher les stats
                st.caption("ℹ️ Réponse générée par Mistral Large via RAG")
                
            except Exception as e:
                st.error(f"Une erreur est survenue: {str(e)}")
    else:
        st.warning("Veuillez entrer une question valide")

# Style CSS personnalisé
st.markdown("""
<style>
pre {
    background-color: #f5f5f5;
    border-radius: 5px;
    padding: 15px;
    overflow-x: auto;
}
code {
    font-family: 'Courier New', monospace;
}
</style>
""", unsafe_allow_html=True)