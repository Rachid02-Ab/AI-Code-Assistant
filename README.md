# 🐍 AI Coder - Chatbot d'assistance Python

![Description de l'image](images/screen-capture.webm)
![Description de l'image](images/autre%20image.png)

AI Coder est un assistant intelligent (**copilot**) capable de répondre à des questions techniques sur le langage Python. Il utilise une architecture **RAG (Retrieval-Augmented Generation)** combinée à un modèle de langage (**Mistral LLM**) pour fournir des réponses précises et contextualisées.

## 🛠️ Installation

1. Clonez le dépôt :
```bash
git clone https://github.com/Rachid02-Ab/AI-Code-Assistant.git
cd AI Coder 
```

## ✨ Fonctionnalités

- Réponses précises avec mise en forme du code
- Explications claires des concepts Python
- Exemples de code exécutables
- Interface conviviale avec Streamlit
- Base de connaissances technique intégrée (HuggingFace dataset)

### 1. Chargement des données (Q&A)
Le chatbot utilise un fichier JSON (`glaive_dataset.json`)  contenant des centaines de questions/réponses sur le code Python, comme **source de connaissance externe**, ces données sont disponibles sur **HuggingFace dataset**.

### 2. Technologies utilisées
- **LangChain** – gestion du pipeline RAG
- **FAISS** – Vector database
- **HuggingFace Embeddings** – conversion texte → vecteur
- **Mistral**  – LLM utilisé dans ce projet
- **Streamlit** – interface utilisateur

## 🔁 CI/CD avec GitHub Actions

Une pipeline CI/CD est configurée pour ce projet afin d'assurer la qualité du code et le bon fonctionnement de l'application à chaque **pull request vers la branche `main`**.

### ✔️ Étapes automatisées :

- 🧹 Analyse du code avec `flake8`
- 🎨 Vérification du formatage avec `black`
- 🚀 Lancement de l'application Streamlit en tâche de fond
- ✅ Vérification que l'interface est accessible via `localhost:8501`
- 📂 Fichier : `.github/workflows/ci.yml`
