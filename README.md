# Projet RAG Local (Retrieval-Augmented Generation)

Pour plus de détails techniques, consultez le fichier `WORKFLOW.md`.

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

3. Configurez les variables d'environnement :
   ```bash
   cp .env.example .env
   # Remplissez FAL_KEY, QDRANT_URL et QDRANT_API_KEY
   ```

## 🛠️ Utilisation

### Etape 1 : Collecte des sources (optionnel)

Télécharge les articles Wikipedia et les papers OpenAlex pour chaque entrée dans `data/dictators.txt` et `data/related_topics.txt`. Les articles déjà téléchargés sont skip automatiquement.

```bash
uv run python -m core.sources
```

*Note : Ne lancez cette commande que si vous ajoutez de nouveaux dictateurs ou sujets dans `data/`. Les articles sont versionnés dans `assets/` et déjà inclus dans le dépôt.*

### Etape 2 : Ingestion dans Qdrant

```bash
uv run python -m core.ingestion
```

*Note : Ne lancez cette commande qu'une seule fois ou à chaque fois que vous ajoutez/modifiez de nouveaux documents dans le dossier assets/.*

### Etape 3 : Lancement de l'interface

Une fois l'ingestion terminée, vous pouvez démarrer l'interface de discussion (Streamlit) :

```bash
uv run chainlit run app.py
```

L'application s'ouvrira automatiquement dans votre navigateur par défaut (généralement à l'adresse `http://localhost:8501`).

### Docker 

```bash
docker build -t rag-app .

# Run
docker run -p 8000:8000 --env-file .env rag-app
```
