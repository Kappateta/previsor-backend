from flask import Flask, jsonify
from flask_cors import CORS
import joblib
import json
import numpy as np

app = Flask(__name__)
CORS(app)

# Carregar modelos
rf  = joblib.load("modelo_random_forest.pkl")
xgb = joblib.load("modelo_xgboost.pkl")
mlp = joblib.load("modelo_rede_neural.pkl")

with open("colunas.json") as f:
    COLUNAS = json.load(f)

LABELS = {0: "Derrota", 1: "Empate", 2: "Vitoria"}

STATS = {
    "mandante": {
        "Flamengo":      {"chutes":7.0,"alvo":2.1,"passes":216.3,"faltas":6.3,"escanteios":2.9,"amarelo":0.8,"vermelho":0.0,"posse":59.3},
        "Palmeiras":     {"chutes":7.2,"alvo":2.3,"passes":310.5,"faltas":5.8,"escanteios":3.1,"amarelo":0.7,"vermelho":0.0,"posse":57.2},
        "Atletico-MG":   {"chutes":6.8,"alvo":2.0,"passes":198.4,"faltas":6.8,"escanteios":2.7,"amarelo":0.9,"vermelho":0.1,"posse":52.1},
        "Botafogo":      {"chutes":6.1,"alvo":1.8,"passes":187.2,"faltas":7.1,"escanteios":2.5,"amarelo":1.0,"vermelho":0.1,"posse":49.8},
        "Corinthians":   {"chutes":6.3,"alvo":1.9,"passes":201.5,"faltas":6.9,"escanteios":2.8,"amarelo":0.9,"vermelho":0.0,"posse":51.3},
        "Sao Paulo":     {"chutes":6.5,"alvo":2.0,"passes":205.8,"faltas":6.5,"escanteios":2.9,"amarelo":0.8,"vermelho":0.0,"posse":53.4},
        "Internacional": {"chutes":6.7,"alvo":2.1,"passes":209.3,"faltas":6.2,"escanteios":2.8,"amarelo":0.8,"vermelho":0.0,"posse":54.1},
        "Fluminense":    {"chutes":6.4,"alvo":1.9,"passes":203.7,"faltas":6.4,"escanteios":2.7,"amarelo":0.9,"vermelho":0.0,"posse":52.8},
        "Santos":        {"chutes":6.2,"alvo":1.8,"passes":195.4,"faltas":6.7,"escanteios":2.6,"amarelo":0.9,"vermelho":0.1,"posse":50.9},
        "Cruzeiro":      {"chutes":6.0,"alvo":1.7,"passes":192.1,"faltas":7.0,"escanteios":2.5,"amarelo":1.0,"vermelho":0.1,"posse":49.5},
        "Vasco":         {"chutes":5.9,"alvo":1.7,"passes":189.3,"faltas":7.2,"escanteios":2.4,"amarelo":1.0,"vermelho":0.1,"posse":48.7},
        "Bahia":         {"chutes":5.8,"alvo":1.6,"passes":185.2,"faltas":7.3,"escanteios":2.3,"amarelo":1.1,"vermelho":0.1,"posse":47.9},
        "Gremio":        {"chutes":6.6,"alvo":2.0,"passes":207.8,"faltas":6.3,"escanteios":2.8,"amarelo":0.8,"vermelho":0.0,"posse":53.7},
        "RB Bragantino": {"chutes":6.4,"alvo":1.9,"passes":204.5,"faltas":6.6,"escanteios":2.7,"amarelo":0.9,"vermelho":0.0,"posse":52.3},
        "Mirassol":      {"chutes":5.5,"alvo":1.5,"passes":175.3,"faltas":7.5,"escanteios":2.2,"amarelo":1.1,"vermelho":0.1,"posse":46.2},
        "Atletico-PR":   {"chutes":6.3,"alvo":1.9,"passes":199.7,"faltas":6.8,"escanteios":2.7,"amarelo":0.9,"vermelho":0.1,"posse":51.1},
        "Chapecoense":   {"chutes":5.4,"alvo":1.5,"passes":172.8,"faltas":7.6,"escanteios":2.1,"amarelo":1.2,"vermelho":0.1,"posse":45.8},
        "Remo":          {"chutes":5.2,"alvo":1.4,"passes":168.5,"faltas":7.8,"escanteios":2.0,"amarelo":1.2,"vermelho":0.1,"posse":44.9},
        "Coritiba":      {"chutes":5.7,"alvo":1.6,"passes":182.4,"faltas":7.4,"escanteios":2.3,"amarelo":1.1,"vermelho":0.1,"posse":47.3},
    },
    "visitante": {
        "Flamengo":      {"chutes":5.3,"alvo":1.6,"passes":205.2,"faltas":6.0,"escanteios":2.3,"amarelo":1.0,"vermelho":0.1,"posse":56.5},
        "Palmeiras":     {"chutes":5.8,"alvo":1.8,"passes":280.3,"faltas":5.5,"escanteios":2.6,"amarelo":0.8,"vermelho":0.0,"posse":54.8},
        "Atletico-MG":   {"chutes":5.4,"alvo":1.6,"passes":188.7,"faltas":6.5,"escanteios":2.2,"amarelo":1.0,"vermelho":0.1,"posse":49.3},
        "Botafogo":      {"chutes":4.8,"alvo":1.4,"passes":175.3,"faltas":6.8,"escanteios":2.0,"amarelo":1.1,"vermelho":0.1,"posse":46.5},
        "Corinthians":   {"chutes":5.0,"alvo":1.5,"passes":190.2,"faltas":6.6,"escanteios":2.3,"amarelo":1.0,"vermelho":0.0,"posse":48.7},
        "Sao Paulo":     {"chutes":5.2,"alvo":1.6,"passes":193.5,"faltas":6.2,"escanteios":2.4,"amarelo":0.9,"vermelho":0.0,"posse":50.6},
        "Internacional": {"chutes":5.4,"alvo":1.7,"passes":197.8,"faltas":5.9,"escanteios":2.3,"amarelo":0.9,"vermelho":0.0,"posse":51.4},
        "Fluminense":    {"chutes":5.1,"alvo":1.5,"passes":191.4,"faltas":6.1,"escanteios":2.2,"amarelo":1.0,"vermelho":0.0,"posse":49.9},
        "Santos":        {"chutes":4.9,"alvo":1.4,"passes":183.6,"faltas":6.4,"escanteios":2.1,"amarelo":1.0,"vermelho":0.1,"posse":48.2},
        "Cruzeiro":      {"chutes":4.7,"alvo":1.3,"passes":180.4,"faltas":6.7,"escanteios":2.0,"amarelo":1.1,"vermelho":0.1,"posse":46.8},
        "Vasco":         {"chutes":4.6,"alvo":1.3,"passes":177.5,"faltas":6.9,"escanteios":1.9,"amarelo":1.1,"vermelho":0.1,"posse":45.9},
        "Bahia":         {"chutes":4.5,"alvo":1.2,"passes":173.8,"faltas":7.0,"escanteios":1.8,"amarelo":1.2,"vermelho":0.1,"posse":45.1},
        "Gremio":        {"chutes":5.3,"alvo":1.6,"passes":195.6,"faltas":6.0,"escanteios":2.3,"amarelo":0.9,"vermelho":0.0,"posse":50.8},
        "RB Bragantino": {"chutes":5.1,"alvo":1.5,"passes":192.3,"faltas":6.3,"escanteios":2.2,"amarelo":1.0,"vermelho":0.0,"posse":49.5},
        "Mirassol":      {"chutes":4.2,"alvo":1.2,"passes":163.4,"faltas":7.2,"escanteios":1.7,"amarelo":1.2,"vermelho":0.1,"posse":43.5},
        "Atletico-PR":   {"chutes":5.0,"alvo":1.5,"passes":187.9,"faltas":6.5,"escanteios":2.2,"amarelo":1.0,"vermelho":0.1,"posse":48.4},
        "Chapecoense":   {"chutes":4.1,"alvo":1.1,"passes":160.5,"faltas":7.3,"escanteios":1.6,"amarelo":1.3,"vermelho":0.1,"posse":43.0},
        "Remo":          {"chutes":3.9,"alvo":1.1,"passes":156.8,"faltas":7.5,"escanteios":1.5,"amarelo":1.3,"vermelho":0.1,"posse":42.1},
        "Coritiba":      {"chutes":4.4,"alvo":1.3,"passes":170.7,"faltas":7.1,"escanteios":1.8,"amarelo":1.2,"vermelho":0.1,"posse":44.6},
    }
}

@app.route("/stats/<time>/<lado>")
def get_stats(time, lado):
    try:
        stats = STATS.get(lado, {}).get(time, {
            "chutes":6.0,"alvo":1.8,"passes":190.0,"faltas":7.0,
            "escanteios":2.5,"amarelo":1.0,"vermelho":0.0,"posse":50.0
        })
        return jsonify({**stats, "time": time, "forma": "", "aproveitamento": 50})
    except Exception as e:
        return jsonify({"erro": str(e)}), 500

@app.route("/prever", methods=["GET"])
def prever():
    try:
        import flask
        args = flask.request.args

        features = {
            "mandante_chutes":           float(args.get("mandante_chutes", 0)),
            "visitante_chutes":          float(args.get("visitante_chutes", 0)),
            "mandante_chutes_no_alvo":   float(args.get("mandante_alvo", 0)),
            "visitante_chutes_no_alvo":  float(args.get("visitante_alvo", 0)),
            "mandante_posse_de_bola":    float(args.get("mandante_posse", 50)),
            "visitante_posse_de_bola":   float(args.get("visitante_posse", 50)),
            "mandante_passes":           float(args.get("mandante_passes", 0)),
            "visitante_passes":          float(args.get("visitante_passes", 0)),
            "mandante_faltas":           float(args.get("mandante_faltas", 0)),
            "visitante_faltas":          float(args.get("visitante_faltas", 0)),
            "mandante_cartao_amarelo":   float(args.get("mandante_amarelo", 0)),
            "visitante_cartao_amarelo":  float(args.get("visitante_amarelo", 0)),
            "mandante_cartao_vermelho":  float(args.get("mandante_vermelho", 0)),
            "visitante_cartao_vermelho": float(args.get("visitante_vermelho", 0)),
            "mandante_escanteios":       float(args.get("mandante_escanteios", 0)),
            "visitante_escanteios":      float(args.get("visitante_escanteios", 0)),
            "aprov_mandante":            float(args.get("aprov_mandante", 0.5)),
            "aprov_visitante":           float(args.get("aprov_visitante", 0.5)),
        }

        X = np.array([[features[c] for c in COLUNAS]])

        pred_rf  = int(rf.predict(X)[0])
        pred_xgb = int(xgb.predict(X)[0])
        pred_mlp = int(mlp.predict(X)[0])

        probs = xgb.predict_proba(X)[0]

        return jsonify({
            "random_forest": LABELS[pred_rf],
            "xgboost":       LABELS[pred_xgb],
            "rede_neural":   LABELS[pred_mlp],
            "prob_vitoria":  round(float(probs[2]) * 100),
            "prob_empate":   round(float(probs[1]) * 100),
            "prob_derrota":  round(float(probs[0]) * 100),
            "vencedor":      LABELS[pred_xgb]
        })

    except Exception as e:
        return jsonify({"erro": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)