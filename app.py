from flask import Flask, jsonify
from flask_cors import CORS
import requests
import joblib
import json
import numpy as np

app = Flask(__name__)
CORS(app)

API_KEY = "6cfc57c7bemsh086366f363fec79p1d7448jsne35048b2856c"
API_HOST = "sofascore.p.rapidapi.com"

HEADERS = {
    "X-RapidAPI-Key": API_KEY,
    "X-RapidAPI-Host": API_HOST
}

TIME_IDS = {
    "Flamengo": 5981,
    "Palmeiras": 1963,
    "Atletico-MG": 1977,
    "Botafogo": 1958,
    "Corinthians": 1957,
    "Sao Paulo": 1981,
    "Internacional": 1966,
    "Fluminense": 1961,
    "Santos": 1968,
    "Cruzeiro": 1954,
    "Vasco": 1974,
    "Bahia": 1955,
    "Gremio": 5926,
    "RB Bragantino": 2248,
    "Mirassol": 7900,
    "Atletico-PR": 1967,
    "Chapecoense": 133,
    "Remo": 4829,
    "Coritiba": 1982,
    "Santos": 1968
}

# Carregar modelos e colunas
rf  = joblib.load("modelo_random_forest.pkl")
xgb = joblib.load("modelo_xgboost.pkl")
mlp = joblib.load("modelo_rede_neural.pkl")

with open("colunas.json") as f:
    COLUNAS = json.load(f)

LABELS = {0: "Derrota", 1: "Empate", 2: "Vitoria"}

def get_stat(items, key, idx):
    for item in items:
        if item.get("key") == key:
            val = item.get("homeValue") if idx == 0 else item.get("awayValue")
            try:
                return float(str(val).replace("%", "") or 0)
            except:
                return 0
    return 0

def buscar_stats_time(team_id):
    url = f"https://{API_HOST}/teams/get-last-matches?teamId={team_id}&pageIndex=0"
    res = requests.get(url, headers=HEADERS)
    eventos = res.json().get("events", [])

    brasileirao = [
        e for e in eventos
        if "brasil" in e.get("tournament", {}).get("slug", "").lower()
    ][:5]

    pontos = 0
    forma = ""
    stats = {"posse": [], "chutes": [], "escanteios": [], "faltas": [], "passes": [], "amarelo": [], "vermelho": []}

    for evento in brasileirao:
        event_id = evento.get("id")
        home_id = evento.get("homeTeam", {}).get("id")
        is_home = home_id == team_id
        idx = 0 if is_home else 1

        gols_pro = evento.get("homeScore", {}).get("current", 0) if is_home else evento.get("awayScore", {}).get("current", 0)
        gols_contra = evento.get("awayScore", {}).get("current", 0) if is_home else evento.get("homeScore", {}).get("current", 0)

        if gols_pro is None or gols_contra is None:
            continue

        if gols_pro > gols_contra: pontos += 3; forma += "V"
        elif gols_pro == gols_contra: pontos += 1; forma += "E"
        else: forma += "D"

        url_stats = f"https://{API_HOST}/matches/get-statistics?matchId={event_id}"
        res_stats = requests.get(url_stats, headers=HEADERS)
        stats_data = res_stats.json().get("statistics", [])
        all_stats = next((s for s in stats_data if s.get("period") == "ALL"), None)
        if not all_stats:
            continue

        items = []
        for group in all_stats.get("groups", []):
            items += group.get("statisticsItems", [])

        stats["posse"].append(get_stat(items, "ballPossession", idx))
        stats["chutes"].append(get_stat(items, "totalShotsOnGoal", idx))
        stats["escanteios"].append(get_stat(items, "cornerKicks", idx))
        stats["faltas"].append(get_stat(items, "fouls", idx))
        stats["passes"].append(get_stat(items, "passes", idx))
        stats["amarelo"].append(get_stat(items, "yellowCards", idx))
        stats["vermelho"].append(get_stat(items, "redCards", idx))

    def media(lst):
        return round(sum(lst) / len(lst), 1) if lst else 0

    aproveitamento = round((pontos / (len(brasileirao) * 3)) * 100) if brasileirao else 50

    return {
        "forma": forma,
        "aproveitamento": aproveitamento,
        "chutes": media(stats["chutes"]),
        "alvo": round(media(stats["chutes"]) * 0.35, 1),
        "passes": media(stats["passes"]),
        "faltas": media(stats["faltas"]),
        "escanteios": media(stats["escanteios"]),
        "amarelo": media(stats["amarelo"]),
        "vermelho": media(stats["vermelho"]),
        "posse": media(stats["posse"])
    }

@app.route("/stats/<time>/<lado>")
def get_stats(time, lado):
    team_id = TIME_IDS.get(time)
    if not team_id:
        return jsonify({"erro": "Time nao encontrado"}), 404
    try:
        data = buscar_stats_time(team_id)
        return jsonify({**data, "time": time})
    except Exception as e:
        return jsonify({"erro": str(e)}), 500

@app.route("/prever", methods=["GET"])
def prever():
    try:
        import flask
        args = flask.request.args

        # Montar vetor de features na mesma ordem do treino
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

        # Probabilidades do XGBoost
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
