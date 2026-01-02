import mlflow
import dagshub

# 1. Connexion automatique à ton dépôt DagsHub
dagshub.init(repo_owner="LatifahFall", repo_name="ProjetAI", mlflow=True)

# 2. On définit l'URL MLflow
mlflow.set_tracking_uri("https://dagshub.com/LatifahFall/ProjetAI.mlflow")

# 3. On crée un "Run" et on enregistre le modèle
with mlflow.start_run(run_name="Upload_Initial_Model"):
    # On logue le fichier (vérifie bien que le chemin vers ton .pt est correct)
    mlflow.log_artifact("models/model_4_best.pt")
    
    # On l'enregistre officiellement dans le Model Registry
    # Cela va créer "Version 1" automatiquement
    result = mlflow.register_model(
        f"runs:/{mlflow.active_run().info.run_id}/model_4_best.pt",
        "ocean-predictor"
    )
    print("Succès ! Le modèle est enregistré dans le Model Registry.")