from flask import Flask, request, jsonify, Blueprint
from abuse_detector import analyze_all_patients, train_model, analyze_patient
import json

app = Flask(__name__)

# Crear un blueprint con prefijo /model
api = Blueprint('model', __name__, url_prefix='/model')

@api.route("/train", methods=["POST"])
def trainModel():
    train_stats = train_model()
    return jsonify(train_stats)

@api.route("/analyze/<userId>", methods=["GET"])
def analyse(userId):
    result = analyze_patient(userId)
    return jsonify(result)

@api.route("/all", methods=["GET"])
def get_all():
    result = analyze_all_patients()
    return jsonify(result)

@api.route("/health", methods=["GET"])
def healt():
    return jsonify({"status":"ok"})

# Registrar el blueprint en la aplicación
app.register_blueprint(api)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
