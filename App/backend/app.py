import os
from flask import Flask, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

# Importation des routes (Blueprints)
# Ces fichiers doivent exister dans le dossier 'routes'
from routes.auth import auth_bp
from routes.predict import predict_bp

# Chargement des variables d'environnement (.env)
load_dotenv()

def create_app():
    app = Flask(__name__)
    
    # Configuration CORS : Permet à React (port 3000) de parler à Flask (port 5000)
    CORS(app, resources={r"/api/*": {"origins": "*"}})

    # --- Enregistrement des Blueprints ---
    # Chaque membre de l'équipe peut travailler sur son propre fichier dans /routes
    
    # Toi (Fille 1) : Gestion du Login
    app.register_blueprint(auth_bp, url_prefix='/api/auth')
    
    # Fille 2 : Gestion de l'audio et du modèle ML
    app.register_blueprint(predict_bp, url_prefix='/api/ml')

    # Route de base pour tester si le serveur fonctionne
    @app.route('/')
    def index():
        return jsonify({
            "status": "success",
            "message": "Backend Flask de l'App Big Five est opérationnel",
            "version": "1.0.0"
        })

    return app

if __name__ == '__main__':
    app = create_app()
    # On lance sur le port 5000 en mode debug pour voir les erreurs en direct
    app.run(host='0.0.0.0', port=5000, debug=True)