# 🐍 AI Coder - Chatbot d'assistance Python

![Description de l'image](images/github%20README.png)
![Description de l'image](images/autre%20image.png)

AI Coder est un assistant intelligent capable de répondre à des questions techniques sur le langage Python. Il utilise une architecture RAG (Retrieval-Augmented Generation) combinée à un modèle de langage (LLM) pour fournir des réponses précises et contextualisées.


## ✨ Fonctionnalités

- Réponses précises avec mise en forme du code
- Explications claires des concepts Python
- Exemples de code exécutables
- Interface conviviale avec Streamlit
- Base de connaissances technique intégrée (HuggingFace dataset)

### 1. Chargement des données (Q&A)
Le chatbot utilise un fichier JSON (`glaive_dataset.json`)  contenant des centaines de questions/réponses sur le code Python, comme source de connaissance externe, ces données sont disponibles sur HuggingFace dataset.

### 2. Technologies utilisées
LangChain – gestion du pipeline RAG

FAISS – base vectorielle locale

HuggingFace Embeddings – conversion texte → vecteur

Mistral  – génération de texte avec LLM

Streamlit – interface utilisateur

## 🛠️ Installation

1. Clonez le dépôt :
```bash
git clone https://github.com/Rachid02-Ab/AI-Code-Assistant.git
cd AI Coder