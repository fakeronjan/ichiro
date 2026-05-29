"""
generate_data.py — reads ichiro_ratings.csv + all_games.csv + tournament_podiums.csv
and writes the JSON the ICHIRO web frontend (a MESSI-clone single-page app)
consumes. Run after ichiro.py. Outputs to docs/data/.

Emits the MESSI JSON contract so the ported index.html works unchanged:
  seasons_index.json      {seasons, first_date, last_date, generated_at}
  seasons/<year>.json     {season, snapshots:[{date, label, prestige, teams:[...]}]}
  teams_index.json        [{name, flag, confederation, slug}, ...]
  teams/<slug>.json       {team, flag, confederation, seasons:{<year>:[rows]}}
  goat_teams.json         top single-snapshot ratings at tournament finals
  medals.json             per-country gold/silver/bronze counts (Medals tab)
  current_standings.json  most-recent snapshot leaderboard (compatibility)

Sport adaptation vs MESSI:
  - confederations are display-only regions: Asia / Americas / Europe / Africa / Oceania
  - margins are "runs" (baseball), no draws — W-L records only
  - "League History" tab is replaced by a "Medals" tab built from medals.json
  - no continental championships in international baseball, so the
    continental-winner gold-pill never fires (always 0)
"""

import json
import os
import re
import pandas as pd
from datetime import datetime, timezone

from ichiro import code_to_country

DOCS = "docs"
DATA = os.path.join(DOCS, "data")
TEAMS = os.path.join(DATA, "teams")
os.makedirs(TEAMS, exist_ok=True)
os.makedirs(os.path.join(DATA, "seasons"), exist_ok=True)

MIN_GAMES = 4       # eligibility for displayed leaderboard
GOAT_MIN_GAMES = 6  # peak must rest on a meaningful body of work

# Tournaments that crown a champion (all four international events).
MEDAL_TOURNAMENTS = ["World Baseball Classic", "Olympics",
                     "WBSC Premier12", "Baseball World Cup"]

# Standings season-file labels: tournament -> (label, prestige). Lower prestige
# = more prestigious (controls dropdown default + ordering), mirroring MESSI.
TOURNAMENT_LABELS = {
    "World Baseball Classic": ("WBC Final",        1),
    "Olympics":               ("Olympic Final",    2),
    "WBSC Premier12":         ("Premier12 Final",  3),
    "Baseball World Cup":     ("World Cup Final",   4),
}

# Tournament name -> short label for honor badges.
TOURNAMENT_ABBREV = {
    "World Baseball Classic": "WBC",
    "Olympics":               "Oly",
    "WBSC Premier12":         "P12",
    "Baseball World Cup":     "WC",
}

# Country -> ISO alpha-2 for flag emoji.
COUNTRY_TO_ISO = {
    "Australia": "AU", "Brazil": "BR", "Canada": "CA", "China": "CN",
    "Colombia": "CO", "Cuba": "CU", "Czech Republic": "CZ",
    "Dominican Republic": "DO", "Spain": "ES", "Great Britain": "GB",
    "Greece": "GR", "Israel": "IL", "Italy": "IT", "Japan": "JP",
    "South Korea": "KR", "Mexico": "MX", "Nicaragua": "NI",
    "Netherlands": "NL", "Panama": "PA", "Puerto Rico": "PR",
    "South Africa": "ZA", "Chinese Taipei": "TW", "United States": "US",
    "Venezuela": "VE",
    # Minor nations that appear in the ratings via raw codes (see CODE_FIXUP)
    "Croatia": "HR", "France": "FR", "Germany": "DE", "Russia": "RU",
    "Sweden": "SE",
    # Netherlands Antilles (AHO) is defunct — no flag emoji.
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
    "Israel": "Europe", "Croatia": "Europe", "France": "Europe",
    "Germany": "Europe", "Russia": "Europe", "Sweden": "Europe",
    "South Africa": "Africa",
    "Netherlands Antilles": "Americas",
}

# A handful of raw 3-letter codes leaked into the ratings `name` column
# (ichiro.py's CODE_TO_COUNTRY didn't map them). Canonicalize here so they
# display as countries with flags/confederations. We do NOT touch the engine.
CODE_FIXUP = {
    "AHO": "Netherlands Antilles", "CRO": "Croatia", "FRA": "France",
    "GER": "Germany", "NIC": "Nicaragua", "RUS": "Russia", "SWE": "Sweden",
}


def canon_name(name):
    return CODE_FIXUP.get(str(name).strip(), str(name).strip())


def flag_emoji(country):
    iso = COUNTRY_TO_ISO.get(canon_name(country))
    if not iso:
        return ""
    return "".join(chr(0x1F1E6 + ord(c) - ord("A")) for c in iso)


def confed_for(country):
    return CONFEDERATION.get(canon_name(country), "Other")


def slug(name):
    return re.sub(r"[^\w]", "_", canon_name(name)).strip("_")


def clean(val):
    if pd.isna(val):
        return ""
    return str(val)


def round2(x):
    try:
        return round(float(x), 2)
    except (TypeError, ValueError):
        return None


def jdump(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, separators=(",", ":"), ensure_ascii=False)


# ============================================================
print("Reading data...")
df = pd.read_csv("ichiro_ratings.csv")
df["date"] = pd.to_datetime(df["ranking_date"]).dt.date
df["name"] = df["name"].map(canon_name)
df["country"] = df["name"]
df["confederation"] = df["name"].map(confed_for)
df["is_game_day"] = 1  # every ichiro snapshot date is a game-day

games = pd.read_csv("all_games.csv")
games["date"] = pd.to_datetime(games["date"]).dt.date
games["home_team"] = games["home_team"].map(code_to_country).map(canon_name)
games["road_team"] = games["road_team"].map(code_to_country).map(canon_name)
games["home_runs"] = pd.to_numeric(games["home_runs"], errors="coerce")
games["road_runs"] = pd.to_numeric(games["road_runs"], errors="coerce")
games = games.dropna(subset=["home_runs", "road_runs"])

podiums = pd.read_csv("tournament_podiums.csv")
podiums["team"] = podiums["team"].map(canon_name)


# ── W-L record per country over full history (no draws in baseball) ──────────
records = {}  # country -> {"w","l"}
for _, g in games.iterrows():
    hw = g["home_runs"] > g["road_runs"]
    for team, won in ((g["home_team"], hw), (g["road_team"], not hw)):
        r = records.setdefault(team, {"w": 0, "l": 0})
        r["w" if won else "l"] += 1


def record_str(name):
    r = records.get(canon_name(name), {"w": 0, "l": 0})
    return f"{r['w']}-{r['l']}"


# ── Last game per (team, date) — the game that produced each snapshot ────────
# For each team, the most recent game on/before a given snapshot date. Built as
# a sorted per-team list so a bisect-style scan resolves the snapshot's row.
team_games = {}  # team -> [(date, "W 6-4 vs. Netherlands (Olympics)")]
for _, g in games.sort_values("date").iterrows():
    for team, opp, rs, oruns in (
        (g["home_team"], g["road_team"], g["home_runs"], g["road_runs"]),
        (g["road_team"], g["home_team"], g["road_runs"], g["home_runs"]),
    ):
        wl = "W" if rs > oruns else "L"
        txt = f"{wl} {int(rs)}-{int(oruns)} vs. {opp} ({g['tournament']})"
        team_games.setdefault(team, []).append((g["date"], txt))


def last_game_asof(team, asof_date):
    """Most recent game text + date for `team` on/before asof_date."""
    best = None
    for d, txt in team_games.get(team, []):
        if d <= asof_date:
            best = (d, txt)
        else:
            break
    return best  # (date, text) or None


# ============================================================
# MEDALS — per-edition gold/silver/bronze come straight from the podium table.
# (Unlike CARMELO we don't have to walk the bracket: ICHIRO ships a curated
# tournament_podiums.csv with finish positions per edition.)
# ============================================================
edition_results = {}  # (tournament, season) -> {1:gold,2:silver,3:bronze}
medal_counts = {}     # country -> {"gold","silver","bronze"}
for (tour, season), grp in podiums.groupby(["tournament", "season"]):
    res = {int(r["finish"]): r["team"] for _, r in grp.iterrows()}
    edition_results[(tour, int(season))] = res
    for finish, name in res.items():
        if not name:
            continue
        m = medal_counts.setdefault(name, {"gold": 0, "silver": 0, "bronze": 0})
        m[{1: "gold", 2: "silver", 3: "bronze"}[finish]] += 1

# Per-(country, year) tournament finishes for honor badges.
country_year_finishes = {}
for (tour, season), res in edition_results.items():
    for finish, name in res.items():
        if not name:
            continue
        country_year_finishes.setdefault((name, season), []).append(
            {"tournament": TOURNAMENT_ABBREV.get(tour, tour), "finish": finish})
for key in country_year_finishes:
    country_year_finishes[key].sort(key=lambda x: x["finish"])


def finishes_for(name, year):
    if pd.isna(year):
        return []
    return country_year_finishes.get((canon_name(name), int(year)), [])


# ── medals.json: per-country aggregated counts for the Medals tab ────────────
print("Writing medals.json...")
medal_rows = []
for name, m in medal_counts.items():
    medal_rows.append({
        "team": name,
        "flag": flag_emoji(name),
        "confederation": confed_for(name),
        "gold": m["gold"], "silver": m["silver"], "bronze": m["bronze"],
        "total": m["gold"] + m["silver"] + m["bronze"],
    })
medal_rows.sort(key=lambda x: (-x["gold"], -x["silver"], -x["bronze"], x["team"]))
jdump(medal_rows, os.path.join(DATA, "medals.json"))
print(f"  medals.json: {len(medal_rows)} countries with podiums")


# ============================================================
# TOURNAMENT FINAL DATES (per tournament+year) — labels, GOAT + year anchors
# ============================================================
tournament_final_date = {}
for tour in MEDAL_TOURNAMENTS:
    tg = games[games["tournament"] == tour]
    if tg.empty:
        continue
    for year, grp in tg.groupby("season"):
        tournament_final_date[(tour, int(year))] = grp["date"].max()

final_dates = set(tournament_final_date.values())
df["is_end_of_season"] = df["date"].apply(lambda d: 1 if d in final_dates else 0)

date_label_map = {}  # date_str -> (label, prestige)
for (tour, year), fdate in tournament_final_date.items():
    lbl, prestige = TOURNAMENT_LABELS.get(tour, (tour, 99))
    ds = str(fdate)
    if ds not in date_label_map or date_label_map[ds][1] > prestige:
        date_label_map[ds] = (lbl, prestige)

# ── Per-(team, year) participation per tournament (year anchors) ─────────────
team_year_tournaments = {}
for tour in MEDAL_TOURNAMENTS:
    tg = games[games["tournament"] == tour]
    for _, g in tg.iterrows():
        yr = int(g["season"])
        for name in (g["home_team"], g["road_team"]):
            team_year_tournaments.setdefault((name, yr), set()).add(tour)

team_year_last_game = (
    df[df["is_game_day"] == 1].dropna(subset=["season"])
      .groupby(["name", "season"])["date"].max().to_dict()
)

# Year-anchor priority: WBC (premier global event) > Olympics > Premier12 >
# Baseball World Cup > last game-day of the year.
ANCHOR_PRIORITY = [
    ("World Baseball Classic", "End of World Baseball Classic"),
    ("Olympics",               "End of Olympic baseball"),
    ("WBSC Premier12",         "End of WBSC Premier12"),
    ("Baseball World Cup",     "End of Baseball World Cup"),
]

team_year_anchor = {}
for (name, year_f), last_game in team_year_last_game.items():
    year = int(year_f)
    played = team_year_tournaments.get((name, year), set())
    chosen = None
    for tour, label in ANCHOR_PRIORITY:
        if tour in played and (tour, year) in tournament_final_date:
            chosen = (tournament_final_date[(tour, year)], label)
            break
    if chosen is None:
        chosen = (last_game, "End of year")
    team_year_anchor[(name, year)] = chosen

df["is_year_anchor"] = 0
df["year_anchor_label"] = ""
for (name, year), (d, label) in team_year_anchor.items():
    mask = (df["name"] == name) & (df["date"] == d) & (df["season"] == year)
    if mask.any():
        df.loc[mask, "is_year_anchor"] = 1
        df.loc[mask, "year_anchor_label"] = label

# ── Attach last_match per row (game that produced the snapshot) ──────────────
def _row_last_match(r):
    lg = last_game_asof(r["name"], r["date"])
    return lg[1] if lg else ""


def _row_last_match_date(r):
    lg = last_game_asof(r["name"], r["date"])
    return str(lg[0]) if lg else ""


df["last_match"] = df.apply(_row_last_match, axis=1)
df["last_match_date"] = df.apply(_row_last_match_date, axis=1)

# ── Rank + conf_rank recomputed within MIN_GAMES-eligible teams per snapshot ──
df["eligible"] = df["games_played"] >= MIN_GAMES
df["rank"] = (
    df[df["eligible"]].groupby("ranking_id")["rating"].rank(method="min", ascending=False)
)
df["conf_rank"] = (
    df[df["eligible"]].groupby(["ranking_id", "confederation"])["rating"]
      .rank(method="min", ascending=False)
)


# ============================================================
# 1) SEASON STANDINGS FILES (one per year) + seasons_index.json
# ============================================================
print("Writing season standings files...")
all_seasons = sorted(int(s) for s in df["season"].dropna().unique())


def team_row(r):
    return {
        "rank":                int(r["rank"]) if not pd.isna(r["rank"]) else None,
        "team":                r["name"],
        "flag":                flag_emoji(r["name"]),
        "confederation":       clean(r["confederation"]),
        "rating":              round2(r["rating"]),
        "record":              record_str(r["name"]),
        "last_match":          clean(r["last_match"]),
        "last_match_date":     clean(r["last_match_date"]),
        "tournament_finishes": finishes_for(r["name"], r["season"]),
        "continental_winner":  0,  # no continental tournaments in intl baseball
    }


for season in all_seasons:
    sdf = df[(df["season"] == season) & df["eligible"]]
    snapshots = []
    for ranking_id, rdf in sdf.groupby("ranking_id"):
        rdf = rdf.sort_values("rank")
        snap_date = str(rdf["date"].iloc[0])
        label, prestige = date_label_map.get(snap_date, (None, None))
        snapshots.append({
            "date": snap_date, "label": label, "prestige": prestige,
            "teams": [team_row(r) for _, r in rdf.iterrows()],
        })
    snapshots.sort(key=lambda x: x["date"])
    jdump({"season": season, "snapshots": snapshots},
          os.path.join(DATA, "seasons", f"{season}.json"))

seasons_meta = {
    "seasons":      list(reversed(all_seasons)),
    "first_date":   str(df["date"].min()),
    "last_date":    str(df["date"].max()),
    "generated_at": datetime.now(timezone.utc).isoformat(),
}
jdump(seasons_meta, os.path.join(DATA, "seasons_index.json"))
print(f"  {len(all_seasons)} season files + seasons_index.json")


# ============================================================
# 2) CURRENT STANDINGS (latest snapshot) — compatibility output
# ============================================================
latest_id = int(df["ranking_id"].max())
latest = df[(df["ranking_id"] == latest_id) & df["eligible"]].sort_values("rank")
latest_date = str(latest["date"].iloc[0]) if len(latest) else seasons_meta["last_date"]
standings = []
for _, r in latest.iterrows():
    row = team_row(r)
    row["games_played"] = int(r["games_played"]) if not pd.isna(r["games_played"]) else 0
    standings.append(row)
jdump({"updated": latest_date, "teams": standings},
      os.path.join(DATA, "current_standings.json"))
print(f"  current_standings.json: {len(standings)} teams as of {latest_date}")


# ============================================================
# 3) GOAT TABLE — top single-snapshot ratings at tournament finals
# ============================================================
print("Writing goat_teams.json...")
eligible_podiums = []  # (name, year, tournament)
for (tour, season), res in edition_results.items():
    for finish, name in res.items():
        if name:
            eligible_podiums.append((name, season, tour))
eligible_podiums = sorted(set(eligible_podiums))

df_idx = df.copy()
df_idx["_date_str"] = df_idx["date"].astype(str)
df_idx = df_idx.set_index(["name", "_date_str"])

goat_candidates = []
for name, year, tour in eligible_podiums:
    fdate = tournament_final_date.get((tour, year))
    if fdate is None:
        continue
    try:
        snap = df_idx.loc[(name, str(fdate))]
    except KeyError:
        continue
    if isinstance(snap, pd.DataFrame):
        snap = snap.iloc[0]
    if pd.isna(snap.get("rating")):
        continue
    if snap.get("games_played", 0) < GOAT_MIN_GAMES:
        continue
    goat_candidates.append({
        "name": name, "year": year, "tournament": tour,
        "rating": float(snap["rating"]),
        "confederation": clean(snap.get("confederation", "")),
    })

if goat_candidates:
    goat_df = (
        pd.DataFrame(goat_candidates)
        .sort_values("rating", ascending=False)
        .drop_duplicates(subset=["name", "year"], keep="first")
        .head(50)
        .reset_index(drop=True)
    )
else:
    goat_df = pd.DataFrame(columns=["name", "year", "tournament", "rating", "confederation"])

goat_data = []
for i, (_, r) in enumerate(goat_df.iterrows()):
    goat_data.append({
        "rank":                i + 1,
        "team":                r["name"],
        "flag":                flag_emoji(r["name"]),
        "confederation":       clean(r["confederation"]),
        "season":              int(r["year"]),
        "rating":              round2(r["rating"]),
        "tournament_finishes": finishes_for(r["name"], r["year"]),
        "continental_winner":  0,
    })
jdump(goat_data, os.path.join(DATA, "goat_teams.json"))
print(f"  goat_teams.json: {len(goat_data)} teams")


# ============================================================
# 4) PER-TEAM JSON FILES + teams_index.json
# ============================================================
print("Writing per-team JSON files...")
team_data = df[(df["is_game_day"] == 1) | (df["is_end_of_season"] == 1) |
               (df["is_year_anchor"] == 1)].copy()
team_data = team_data.sort_values(["name", "date"])

played_team_years = set(
    (n, int(y)) for n, y in
    df.loc[df["is_game_day"] == 1, ["name", "season"]].dropna().itertuples(index=False, name=None)
)

all_names = sorted(df["name"].unique())
teams_index = []
for name in all_names:
    tdf = team_data[team_data["name"] == name]
    if len(tdf) == 0:
        continue
    team_slug = slug(name)
    confed = confed_for(name)
    fl = flag_emoji(name)
    teams_index.append({"name": name, "flag": fl, "confederation": confed, "slug": team_slug})

    seasons = {}
    for season, sdf in tdf.groupby("season"):
        if pd.isna(season):
            continue
        if (name, int(season)) not in played_team_years:
            continue
        fin = finishes_for(name, season)
        seasons[int(season)] = [
            {
                "date":                str(r["date"]),
                "rating":              round2(r["rating"]),
                "rank":                int(r["rank"]) if not pd.isna(r["rank"]) else None,
                "conf_rank":           int(r["conf_rank"]) if not pd.isna(r["conf_rank"]) else None,
                "last_match":          clean(r["last_match"]),
                "is_end_of_season":    int(r["is_end_of_season"]),
                "is_game_day":         int(r["is_game_day"]),
                "is_year_anchor":      int(r.get("is_year_anchor", 0) or 0),
                "year_anchor_label":   clean(r.get("year_anchor_label", "")),
                "tournament_finishes": fin,
                "continental_winner":  0,
            }
            for _, r in sdf.sort_values("date").iterrows()
        ]

    jdump({"team": name, "flag": fl, "confederation": confed, "seasons": seasons},
          os.path.join(TEAMS, f"{team_slug}.json"))

teams_index.sort(key=lambda x: x["name"])
jdump(teams_index, os.path.join(DATA, "teams_index.json"))
print(f"  teams_index.json + {len(teams_index)} team files")


# ============================================================
# 5) META (coverage + refresh timestamp)
# ============================================================
meta = {
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "current_date": latest_date,
    "first_date":   str(games["date"].min()),
    "last_date":    str(games["date"].max()),
    "total_games":  int(len(games)),
    "n_tournaments": int(games["tournament"].nunique()),
}
jdump(meta, os.path.join(DATA, "meta.json"))

print(f"Done. {len(teams_index)} teams, {len(standings)} in current standings.")
print(f"Wrote {len(all_seasons)} season files. Standings date: {latest_date}")
