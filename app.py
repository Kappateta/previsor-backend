from flask import Flask, jsonify
from flask_cors import CORS
import requests

app = Flask(__name__)
CORS(app)

API_KEY = "6cfc57c7bemsh086366f363fec79p1d7448jsne35048b2856c"
API_HOST = "sofascore.p.rapidapi.com"

HEADERS = {
    "X-RapidAPI-Key": API_KEY,
    "X-RapidAPI-Host": API_HOST
}

TIME_IDS = {
    "Flamengo": 5981, "Palmeiras": 1963, "Atletico-MG": 1977,
    "Gremio": 5926, "Corinthians": 1957, "Sao Paulo": 1981,
    "Internacional": 1966, "Fluminense": 1961, "Santos": 1968,
    "Botafogo": 1958, "Cruzeiro": 1954, "Atletico-PR": 1967,
    "Vasco": 1974, "Bahia": 1955, "Fortaleza": 2020,
    "Ceara": 2001, "Sport": 1959, "Goias": 1960,
    "Coritiba": 1982, "America-MG": 1973
}

def get_stat(items, key, idx):
    for item in items:
        if item.get("key") == key:
            val = item.get("homeValue") if idx == 0 else item.get("awayValue")
            try:
                return float(str(val).replace("%", "") or 0)
            except:
                return 0
    return 0

@app.route("/stats/<time>/<lado>")
def get_stats(time, lado):
    team_id = TIME_IDS.get(time)
    if not team_id:
        return jsonify({"erro": "Time nao encontrado"}), 404

    try:
        # Buscar ultimas partidas
        url = f"https://{API_HOST}/teams/get-last-matches?teamId={team_id}&pageIndex=0"
        res = requests.get(url, headers=HEADERS)
        eventos = res.json().get("events", [])

        # Filtrar so Brasileirao
        brasileirao = [
            e for e in eventos
            if "brasil" in e.get("tournament", {}).get("slug", "").lower()
        ][:5]

        pontos = 0
        forma = ""
        stats_acumuladas = {
            "posse": [], "chutes": [], "escanteios": [],
            "faltas": [], "passes": [], "amarelo": []
        }

        for evento in brasileirao:
            event_id = evento.get("id")
            home_id = evento.get("homeTeam", {}).get("id")
            is_home = home_id == team_id
            idx = 0 if is_home else 1

            gols_pro = evento.get("homeScore", {}).get("current", 0) if is_home else evento.get("awayScore", {}).get("current", 0)
            gols_contra = evento.get("awayScore", {}).get("current", 0) if is_home else evento.get("homeScore", {}).get("current", 0)

            if gols_pro is None or gols_contra is None:
                continue

            if gols_pro > gols_contra:
                pontos += 3; forma += "V"
            elif gols_pro == gols_contra:
                pontos += 1; forma += "E"
            else:
                forma += "D"

            # Buscar estatísticas da partida
            url_stats = f"https://{API_HOST}/matches/get-statistics?matchId={event_id}"
            res_stats = requests.get(url_stats, headers=HEADERS)
            stats_data = res_stats.json().get("statistics", [])

            all_stats = next((s for s in stats_data if s.get("period") == "ALL"), None)
            if not all_stats:
                continue

            items = []
            for group in all_stats.get("groups", []):
                items += group.get("statisticsItems", [])

            stats_acumuladas["posse"].append(get_stat(items, "ballPossession", idx))
            stats_acumuladas["chutes"].append(get_stat(items, "totalShotsOnGoal", idx))
            stats_acumuladas["escanteios"].append(get_stat(items, "cornerKicks", idx))
            stats_acumuladas["faltas"].append(get_stat(items, "fouls", idx))
            stats_acumuladas["passes"].append(get_stat(items, "passes", idx))
            stats_acumuladas["amarelo"].append(get_stat(items, "yellowCards", idx))

        def media(lst):
            return round(sum(lst) / len(lst), 1) if lst else 0

        aproveitamento = round((pontos / (len(brasileirao) * 3)) * 100) if brasileirao else 50

        return jsonify({
            "time": time,
            "forma": forma,
            "aproveitamento": aproveitamento,
            "chutes": media(stats_acumuladas["chutes"]),
            "alvo": round(media(stats_acumuladas["chutes"]) * 0.35, 1),
            "passes": media(stats_acumuladas["passes"]),
            "faltas": media(stats_acumuladas["faltas"]),
            "escanteios": media(stats_acumuladas["escanteios"]),
            "amarelo": media(stats_acumuladas["amarelo"]),
            "vermelho": 0,
            "posse": media(stats_acumuladas["posse"])
        })

    except Exception as e:
        return jsonify({"erro": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)