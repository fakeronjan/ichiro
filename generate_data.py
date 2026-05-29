"""
generate_data.py - reads ichiro_ratings.csv + all_games.csv + tournament_podiums.csv
and writes JSON for the ICHIRO web frontend. Run after ichiro.py. Outputs to docs/data/.

Mirrors the fleet site architecture (MESSI / GRIFFEY): a current-standings table,
per-team summary with rating history, a champions/podium tab, and a GOAT (peak-
rating) table. International baseball is sparse (one snapshot per game-day across
WBC / Premier12 / Olympics), so "current standings" = the most recent snapshot and
each team's history is a short series of tournament-time ratings.
"""

import json
import os
import re
import pandas as pd
import numpy as np
from datetime import datetime, timezone

DOCS = "docs"
DATA = os.path.join(DOCS, "data")
TEAMS = os.path.join(DATA, "teams")
os.makedirs(TEAMS, exist_ok=True)

# 3-letter-code -> ISO alpha-2 for flag emoji (countries in our dataset only).
COUNTRY_TO_ISO = {
    "Australia": "AU", "Brazil": "BR", "Canada": "CA", "China": "CN",
    "Colombia": "CO", "Cuba": "CU", "Czech Republic": "CZ",
    "Dominican Republic": "DO", "Spain": "ES", "Great Britain": "GB",
    "Greece": "GR", "Israel": "IL", "Italy": "IT", "Japan": "JP",
    "South Korea": "KR", "Mexico": "MX", "Nicaragua": "NI",
    "Netherlands": "NL", "Panama": "PA", "Puerto Rico": "PR",
    "South Africa": "ZA", "Chinese Taipei": "TW", "United States": "US",
    "Venezuela": "VE",
}

CONFEDERATION = {
    "Japan": "Asia", "South Korea": "Asia", "Chinese Taipei": "Asia",
    "China": "Asia", "Australia": "Oceania",
    "United States": "Americas", "Dominican Republic": "Americas",
    "Venezuela": "Americas", "Puerto Rico": "Americas", "Cuba": "Americas",
    "Mexico": "Americas", "Canada": "Americas", "Panama": "Americas",
    "Nicaragua": "Americas", "Colombia": "Americas", "Brazil": "Americas",
    "Netherlands": "Europe", "Italy": "Europe", "Spain": "Europe",
    "Czech Republic": "Europe", "Great Britain": "Europe", "Greece": "Europe",
    "Israel": "Europe", "South Africa": "Africa",
}

# Taiwan has no standard emoji on some platforms; falls back to white flag.
def flag_emoji(country):
    iso = COUNTRY_TO_ISO.get(country)
    if not iso:
        return ""
    return "".join(chr(0x1F1E6 + ord(c) - ord("A")) for c in iso)


def slug(name):
    return re.sub(r"[^\w]", "_", name).strip("_")


def jdump(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, separators=(",", ":"))


# ============================================================
print("Reading data...")
ratings = pd.read_csv("ichiro_ratings.csv")
ratings["ranking_date"] = pd.to_datetime(ratings["ranking_date"]).dt.date
games = pd.read_csv("all_games.csv")
games["date"] = pd.to_datetime(games["date"]).dt.date
podiums = pd.read_csv("tournament_podiums.csv")

# Map game-team codes -> canonical country (same map ichiro.py uses)
from ichiro import code_to_country
games["home_team"] = games["home_team"].map(code_to_country)
games["road_team"] = games["road_team"].map(code_to_country)

LATEST_ID = int(ratings["ranking_id"].max())
LATEST_DATE = ratings.loc[ratings["ranking_id"] == LATEST_ID, "ranking_date"].iloc[0]

# ── Per-team game record (W-L) over full history ─────────────────────────────
def team_records():
    rec = {}
    for _, g in games.iterrows():
        hw = g["home_runs"] > g["road_runs"]
        for team, won in ((g["home_team"], hw), (g["road_team"], not hw)):
            r = rec.setdefault(team, {"w": 0, "l": 0})
            r["w" if won else "l"] += 1
    return rec

RECORDS = team_records()

# ── Last game played per team (most recent across all editions) ──────────────
def last_games():
    out = {}
    for _, g in games.sort_values("date").iterrows():
        for team, opp, rs, os_, venue in (
            (g["home_team"], g["road_team"], g["home_runs"], g["road_runs"], "vs."),
            (g["road_team"], g["home_team"], g["road_runs"], g["home_runs"], "vs."),
        ):
            wl = "W" if rs > os_ else "L"
            out[team] = {
                "date": str(g["date"]),
                "text": f"{wl} {int(rs)}-{int(os_)} {venue} {opp} ({g['tournament']})",
                "wl": wl,
            }
    return out

LAST = last_games()

# ── Medal counts per team ────────────────────────────────────────────────────
medal_counts = {}
for _, p in podiums.iterrows():
    m = medal_counts.setdefault(p["team"], {"gold": 0, "silver": 0, "bronze": 0})
    m[{1: "gold", 2: "silver", 3: "bronze"}[int(p["finish"])]] += 1

# ============================================================
# 1) CURRENT STANDINGS (latest snapshot)
# ============================================================
latest = ratings[ratings["ranking_id"] == LATEST_ID].sort_values("rank")
standings = []
for _, row in latest.iterrows():
    name = row["name"]
    rec = RECORDS.get(name, {"w": 0, "l": 0})
    lg = LAST.get(name, {})
    medals = medal_counts.get(name, {"gold": 0, "silver": 0, "bronze": 0})
    standings.append({
        "rank": int(row["rank"]),
        "name": name,
        "flag": flag_emoji(name),
        "confederation": CONFEDERATION.get(name, "Other"),
        "rating": round(float(row["rating"]), 2),
        "games_played": int(row["games_played"]),
        "record": f"{rec['w']}-{rec['l']}",
        "gold": medals["gold"], "silver": medals["silver"], "bronze": medals["bronze"],
        "last_match": lg.get("text", ""),
        "last_match_date": lg.get("date", ""),
        "slug": slug(name),
    })
jdump({"as_of": str(LATEST_DATE), "teams": standings}, os.path.join(DATA, "current_standings.json"))
print(f"  current_standings.json: {len(standings)} teams")

# Peak-eligibility floor (see GOAT section): a "peak" must rest on a meaningful
# body of work, not a 3-4 game opening hot streak. Tunable.
PEAK_MIN_GAMES = 6


def peak_snapshot(tr):
    """Highest-rating snapshot resting on >= PEAK_MIN_GAMES games (fall back to
    all snapshots if the team never reached the floor)."""
    elig = tr[tr["games_played"] >= PEAK_MIN_GAMES]
    if not len(elig):
        elig = tr
    return elig.loc[elig["rating"].idxmax()]


# ============================================================
# 2) TEAMS INDEX + per-team detail
# ============================================================
teams_index = []
all_teams = sorted(set(ratings["name"]))
for name in all_teams:
    tr = ratings[ratings["name"] == name].sort_values("ranking_id")
    rec = RECORDS.get(name, {"w": 0, "l": 0})
    medals = medal_counts.get(name, {"gold": 0, "silver": 0, "bronze": 0})
    peak = peak_snapshot(tr)
    cur = tr[tr["ranking_id"] == LATEST_ID]
    cur_rank = int(cur["rank"].iloc[0]) if len(cur) else None
    cur_rating = round(float(cur["rating"].iloc[0]), 2) if len(cur) else None

    teams_index.append({
        "name": name, "slug": slug(name), "flag": flag_emoji(name),
        "confederation": CONFEDERATION.get(name, "Other"),
        "current_rank": cur_rank, "current_rating": cur_rating,
        "peak_rating": round(float(peak["rating"]), 2),
        "record": f"{rec['w']}-{rec['l']}",
        "gold": medals["gold"], "silver": medals["silver"], "bronze": medals["bronze"],
    })

    # Per-team history: one point per snapshot where the team appears.
    history = [{
        "date": str(r["ranking_date"]), "season": int(r["season"]),
        "rating": round(float(r["rating"]), 2), "rank": int(r["rank"]),
    } for _, r in tr.iterrows()]

    # Team's games (chronological)
    tg = games[(games["home_team"] == name) | (games["road_team"] == name)].sort_values("date")
    game_log = []
    for _, g in tg.iterrows():
        is_home = g["home_team"] == name
        opp = g["road_team"] if is_home else g["home_team"]
        rs = g["home_runs"] if is_home else g["road_runs"]
        os_ = g["road_runs"] if is_home else g["home_runs"]
        game_log.append({
            "date": str(g["date"]), "tournament": g["tournament"], "season": int(g["season"]),
            "opponent": opp, "opp_flag": flag_emoji(opp),
            "runs_for": int(rs), "runs_against": int(os_),
            "result": "W" if rs > os_ else "L",
            "round": g.get("round", "") if pd.notna(g.get("round", "")) else "",
        })

    jdump({
        "name": name, "flag": flag_emoji(name),
        "confederation": CONFEDERATION.get(name, "Other"),
        "record": f"{rec['w']}-{rec['l']}",
        "gold": medals["gold"], "silver": medals["silver"], "bronze": medals["bronze"],
        "peak_rating": round(float(peak["rating"]), 2),
        "peak_date": str(peak["ranking_date"]),
        "current_rank": cur_rank, "current_rating": cur_rating,
        "history": history, "games": game_log,
    }, os.path.join(TEAMS, f"{slug(name)}.json"))

teams_index.sort(key=lambda t: (t["current_rank"] is None, t["current_rank"] or 999))
jdump(teams_index, os.path.join(DATA, "teams_index.json"))
print(f"  teams_index.json + {len(all_teams)} team files")

# ============================================================
# 3) CHAMPIONS / PODIUMS (per edition)
# ============================================================
finish_label = {1: "Gold", 2: "Silver", 3: "Bronze"}
editions = []
for (tournament, season), grp in podiums.groupby(["tournament", "season"]):
    p = {int(r["finish"]): r["team"] for _, r in grp.iterrows()}
    editions.append({
        "tournament": tournament, "season": int(season),
        "gold": p.get(1, ""), "gold_flag": flag_emoji(p.get(1, "")),
        "silver": p.get(2, ""), "silver_flag": flag_emoji(p.get(2, "")),
        "bronze": p.get(3, ""), "bronze_flag": flag_emoji(p.get(3, "")),
    })
editions.sort(key=lambda e: (e["season"], e["tournament"]), reverse=True)
jdump(editions, os.path.join(DATA, "champions.json"))
print(f"  champions.json: {len(editions)} editions")

# ============================================================
# 4) GOAT TABLE (peak rating, all-time)
# ============================================================
# One row per team: the single highest rating they ever reached, with the
# tournament context (the edition active at that peak date) + medal totals.
#
# PEAK_MIN_GAMES guard (fleet lesson - ghost/early-window snapshots): a peak
# taken from a 3-4 game window rewards a team that merely won its opening pool
# games (Team Israel 2017, Canada 2004 went 4.84 / 5.62 off tiny samples). The
# peak_snapshot() helper (defined above) requires >= PEAK_MIN_GAMES games so the
# GOAT table reflects a real body of work, not an opening hot streak.
goat = []
for name in all_teams:
    tr = ratings[ratings["name"] == name]
    peak = peak_snapshot(tr)
    peak_date = peak["ranking_date"]
    # which edition was live at the peak date?
    live = games[(games["date"] <= peak_date)].sort_values("date")
    edition = ""
    if len(live):
        last_g = live.iloc[-1]
        edition = f"{int(last_g['season'])} {last_g['tournament']}"
    rec = RECORDS.get(name, {"w": 0, "l": 0})
    medals = medal_counts.get(name, {"gold": 0, "silver": 0, "bronze": 0})
    goat.append({
        "name": name, "flag": flag_emoji(name),
        "confederation": CONFEDERATION.get(name, "Other"),
        "peak_rating": round(float(peak["rating"]), 2),
        "peak_date": str(peak_date), "peak_edition": edition,
        "record": f"{rec['w']}-{rec['l']}",
        "gold": medals["gold"], "silver": medals["silver"], "bronze": medals["bronze"],
        "slug": slug(name),
    })
goat.sort(key=lambda g: g["peak_rating"], reverse=True)
for i, g in enumerate(goat, 1):
    g["rank"] = i
jdump(goat, os.path.join(DATA, "goat_teams.json"))
print(f"  goat_teams.json: {len(goat)} teams")

# ============================================================
# 5) META (coverage + refresh timestamp)
# ============================================================
total_games = len(games)
by_tournament = games.groupby("tournament").size().to_dict()
coverage = f"{int(games['season'].min())}-{int(games['season'].max())}"
jdump({
    "as_of": str(LATEST_DATE),
    "coverage": coverage,
    "total_games": int(total_games),
    "by_tournament": {k: int(v) for k, v in by_tournament.items()},
    "refreshed_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
}, os.path.join(DATA, "meta.json"))
print(f"  meta.json: {total_games} games, coverage {coverage}")
print("Done.")
