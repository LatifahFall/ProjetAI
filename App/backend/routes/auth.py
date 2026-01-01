from flask import Blueprint, jsonify

auth_bp = Blueprint('auth', __name__)

@auth_bp.route('/status', methods=['GET'])
def get_status():
    # Cette route permettra au Frontend de vérifier si le serveur est en ligne
    return jsonify({
        "status": "online",
        "message": "Système d'Analyse OCEAN opérationnel",
        "agent_connected": False
    }), 200