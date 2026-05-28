# Projet RAG Local (Retrieval-Augmented Generation)

Pour plus de détails techniques, consultez le fichier `EXPLICATION_PROJET.txt`.

## Prérequis

**uv** : Le gestionnaire de paquets et d'environnements Python ultra-rapide.
   - Installation (Linux/macOS) : `curl -LsSf https://astral.sh/uv/install.sh | sh`
   - Installation (Windows) : `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`

## Installation

1. Clonez ce dépôt ou placez-vous dans le dossier du projet :
   ```bash
   cd /home/maxime/rag
   ```

2. Créez un environnement virtuel et installez les dépendances via `uv` :
   ```bash
   uv sync
   ```

## 🛠️ Utilisation

### Étape 1 : Ingestion des documents

Placez vos documents PDF (et autres extensions si prises en charge) dans le dossier `assets/`. Puis lancez le script d'ingestion pour les analyser et créer la base de données vectorielle locale. Avec `uv`, exécutez :

```bash
uv run python -m core.ingestion
```

*Note : Ne lancez cette commande qu'une seule fois ou à chaque fois que vous ajoutez/modifiez de nouveaux documents dans le dossier assets/.*

### Étape 2 : Lancement de l'interface

Une fois l'ingestion terminée, vous pouvez démarrer l'interface de discussion (Streamlit) :

```bash
uv run streamlit run app.py
# or 
streamlit run app.py
```

L'application s'ouvrira automatiquement dans votre navigateur par défaut (généralement à l'adresse `http://localhost:8501`).

### Docker 

```bash
# Charge d'abord FAL_KEY puis lance le Build
source .env && export $(cut -d= -f1 .env) && docker build -t rag-app --build-arg FAL_KEY=$FAL_KEY .

# Run the container
docker run -p 8501:8501 --env-file .env rag-app
```