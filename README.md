# Projet MLOps - Prédiction de Personnalité

Infrastructure MLOps/DevOps pour le projet de prédiction de personnalité à partir d'audio et texte.

## Structure du Projet

```
projet/
├── mlops/          # Code DevOps (orchestration, pipelines)
├── models/          # Stockage des modèles versionnés
├── mlruns/          # MLflow tracking (gitignored)
├── logs/            # Logs de l'application (gitignored)
├── config/          # Fichiers de configuration
├── config.py        # Configuration centralisée
├── .env.example     # Template pour variables d'environnement
├── requirements.txt  # Dépendances Python
└── README.md        # Ce fichier
```

## Installation

1. Créer un environnement virtuel :
```bash
python -m venv venv
source venv/bin/activate  # Sur Windows: venv\Scripts\activate
```

2. Installer les dépendances :
```bash
pip install -r requirements.txt
```

3. Copier `.env.example` vers `.env` et configurer :
```bash
cp .env.example .env
# Éditer .env avec vos valeurs
```

## Configuration

Les variables d'environnement sont définies dans `.env` et chargées via `config.py`.

## Équipe

- **Latifah** : Infrastructure/Pipeline (Git/Docker/Cloud)
- **Samia** : Backend (API et Traitement audio)
- **Noufissa** : MLflow/Frontend (Interface utilisateur)

## Phases

Voir `PLAN_MLOPS_DEVOPS.md` pour le plan complet.

