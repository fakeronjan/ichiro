"""
generate_data.py — reads ichiro_ratings.csv + all_games.csv + tournament_podiums.csv
and writes the JSON the ICHIRO web frontend (a MESSI-clone single-page app)
consumes. Run after ichiro.py. Outputs to docs/data/.

Emits the MESSI JSON contract so the ported index.html works unchanged:
  seasons_index.json      {seasons, first_date, last_date, generated_at}
  seasons/<year>.json     {season, snapshots:[{date, label, prestige, teams:[...]}]}
  teams_index.json        [{name, flag, confederation, slug}, ...]
  teams/<slug>.json       {team, flag, confederation, seasons:{<year>:[rows]}}
  goat_teams.json         top single-snapshot ratings at FLAGSHIP finals
  champions.json          per-tournament edition podiums (Tournaments tab)
  current_standings.json  most-recent snapshot leaderboard (compatibility)

Sport adaptation vs MESSI:
  - confederations are display-only regions: Asia / Americas / Europe / Africa / Oceania
  - margins are "runs" (baseball), no draws — W-L records only
  - "League History" tab becomes a "Tournaments" tab built from champions.json,
    one row per edition (Gold / Silver / Bronze) with cumulative medal counts,
    each medalist's rating/rank at that edition's final snapshot, and that
    team's W-L within that specific edition.
  - all four global events appear under one Global pill group (no continental
    championships in international baseball); the continental-winner gold-pill
    never fires (always 0).
"""

import json
import os
import re
import bisect
import pandas as pd
from datetime import datetime, timezone, timedelta

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

# Flagship tournaments anchor GOAT entries. 👑 = World Baseball Classic (the
# premier global championship for baseball); the Olympics is the other flagship.
# Premier12 + Baseball World Cup still feed the rolling rating but do NOT anchor
# a GOAT entry.
FLAGSHIP_TOURNAMENTS = ["World Baseball Classic", "Olympics"]

# Tournament selector grouping for the Tournaments tab pills. International
# baseball has no continental championships, so all four events are Global.
GLOBAL_TOURNAMENTS = ["World Baseball Classic", "Olympics",
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


# ── W-L record over the rating WINDOW, AS OF a snapshot date (no draws) ───────
# Each snapshot shows its team's W-L over the WINDOW_YEARS ending at THAT
# snapshot's date, so a historical season file shows the record that produced
# its rating — not a global present-day window (which would, e.g., show a
# nation as 0-0 in an old file just because it has no recent games).
# WINDOW_YEARS must match ichiro.py's window.
WINDOW_YEARS = 4
_WINDOW_DELTA = timedelta(days=int(WINDOW_YEARS * 365.25))
_team_outcomes = {}  # canon name -> ([sorted dates], [won bools aligned])
_tmp_oc = {}
for _, g in games.iterrows():
    hw = g["home_runs"] > g["road_runs"]
    for team, won in ((g["home_team"], hw), (g["road_team"], not hw)):
        _tmp_oc.setdefault(team, []).append((g["date"], bool(won)))
for _c, _lst in _tmp_oc.items():
    _lst.sort(key=lambda x: x[0])
    _team_outcomes[_c] = ([d for d, _ in _lst], [w for _, w in _lst])


def record_str(name, as_of):
    """W-L over the WINDOW_YEARS calendar window ending at as_of (a date)."""
    entry = _team_outcomes.get(canon_name(name))
    if not entry:
        return "0-0"
    dates, wons = entry
    lo = bisect.bisect_left(dates, as_of - _WINDOW_DELTA)
    hi = bisect.bisect_right(dates, as_of)
    w = sum(1 for x in wons[lo:hi] if x)
    return f"{w}-{(hi - lo) - w}"


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
# PODIUMS — per-edition gold/silver/bronze come straight from the podium table.
# (Unlike CARMELO we don't have to walk the bracket: ICHIRO ships a curated
# tournament_podiums.csv with finish positions per edition.)
# ============================================================
edition_results = {}  # (tournament, season) -> {1:gold,2:silver,3:bronze}
for (tour, season), grp in podiums.groupby(["tournament", "season"]):
    res = {int(r["finish"]): r["team"] for _, r in grp.iterrows()}
    edition_results[(tour, int(season))] = res


# ── Per-team W-L within a single tournament edition (no draws in baseball) ────
# Pre-group the games once so each lookup is a dict hit.
edition_groups = {}  # (tournament, season) -> edition game DataFrame
for (tour, season), grp in games[games["tournament"].isin(MEDAL_TOURNAMENTS)].groupby(
        ["tournament", "season"]):
    edition_groups[(tour, int(season))] = grp


def edition_team_wl(tour, season, name):
    """W-L for `name` within one (tournament, season) edition."""
    grp = edition_groups.get((tour, int(season)))
    if grp is None:
        return "0-0"
    w = l = 0
    for _, x in grp.iterrows():
        if x["home_team"] == name:
            won = x["home_runs"] > x["road_runs"]
        elif x["road_team"] == name:
            won = x["road_runs"] > x["home_runs"]
        else:
            continue
        if won:
            w += 1
        else:
            l += 1
    return f"{w}-{l}"

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


# ============================================================
# TOURNAMENT FINAL DATES (per tournament+year) — labels, GOAT + year anchors
# ============================================================
# In-progress gate (see [[feedback-in-progress-season-gate]]): only assign a
# "final date" to a tournament edition that has actually concluded. A naive
# grp["date"].max() labels an in-progress tournament's most recent game-day
# as the "Final", which is wrong. Signals accepted:
#   1. A game with round=='final' or round=='bronze' (medal games played
#      -> tournament concluded).
#   2. A curated podium entry for the edition (the podium table is only
#      populated for concluded events).
podium_editions = set(
    zip(podiums["tournament"], podiums["season"].astype(int))
)
tournament_final_date = {}
for tour in MEDAL_TOURNAMENTS:
    tg = games[games["tournament"] == tour]
    if tg.empty:
        continue
    for year, grp in tg.groupby("season"):
        year = int(year)
        if "round" in grp.columns:
            medal_dates = grp.loc[grp["round"].isin(["final", "bronze"]), "date"]
        else:
            medal_dates = []
        if len(medal_dates):
            tournament_final_date[(tour, year)] = medal_dates.max()
        elif (tour, year) in podium_editions:
            tournament_final_date[(tour, year)] = grp["date"].max()
        # else: in progress -- no Final label assigned

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


def team_row(r, as_of):
    return {
        "rank":                int(r["rank"]) if not pd.isna(r["rank"]) else None,
        "team":                r["name"],
        "flag":                flag_emoji(r["name"]),
        "confederation":       clean(r["confederation"]),
        "rating":              round2(r["rating"]),
        "record":              record_str(r["name"], as_of),
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
        snap_date_obj = rdf["date"].iloc[0]
        snap_date = str(snap_date_obj)
        label, prestige = date_label_map.get(snap_date, (None, None))
        snapshots.append({
            "date": snap_date, "label": label, "prestige": prestige,
            "teams": [team_row(r, snap_date_obj) for _, r in rdf.iterrows()],
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
latest_date_obj = latest["date"].iloc[0] if len(latest) else None
latest_date = str(latest_date_obj) if latest_date_obj is not None else seasons_meta["last_date"]
standings = []
for _, r in latest.iterrows():
    row = team_row(r, latest_date_obj)
    row["games_played"] = int(r["games_played"]) if not pd.isna(r["games_played"]) else 0
    standings.append(row)
jdump({"updated": latest_date, "teams": standings},
      os.path.join(DATA, "current_standings.json"))
print(f"  current_standings.json: {len(standings)} teams as of {latest_date}")


# ============================================================
# 3) GOAT TABLE — top single-snapshot ratings at FLAGSHIP finals
# ============================================================
print("Writing goat_teams.json...")
# Eligibility: medaled (1st/2nd/3rd) in a FLAGSHIP tournament (WBC or Olympics)
# that year. Anchor the rating at THAT tournament's final date. Premier12 +
# Baseball World Cup still feed the rolling rating but do NOT anchor a GOAT
# entry — this dedupes to one entry per team per flagship edition and drops the
# overlapping-window clusters from non-flagship years. Mirrors MESSI's logic.
eligible_podiums = []  # (name, year, tournament)
for (tour, season), res in edition_results.items():
    if tour not in FLAGSHIP_TOURNAMENTS:
        continue
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
# 4b) CHAMPIONS TABLE (per tournament edition) — Tournaments tab
# ============================================================
# Emits the MESSI champions.json contract, grouped by tournament, editions
# newest-first. Each medalist cell carries:
#   - cumulative medal count for that country in that tournament
#   - rating / rank / conf_rank at that edition's FINAL snapshot
#   - W-L within that specific edition (ICHIRO addition vs MESSI)
# Editions where no medalist has a rated snapshot at the final date are marked
# pre_rated (UI renders dashes + † footnote, mirroring MESSI).
print("Writing champions.json...")

# Final-day rating/rank lookup keyed by (name, date_str) — reuse the GOAT index.
_df_str = df.copy()
_df_str["_date_str"] = _df_str["date"].astype(str)
_champ_idx = _df_str.set_index(["name", "_date_str"])


def edition_team_info(name, tour, year):
    """Rating/rank/conf_rank for a medalist at the edition's final snapshot."""
    fdate = tournament_final_date.get((tour, int(year)))
    if fdate is None:
        return {"rating": None, "rank": None, "conf_rank": None, "confederation": confed_for(name)}
    try:
        snap = _champ_idx.loc[(name, str(fdate))]
    except KeyError:
        return {"rating": None, "rank": None, "conf_rank": None, "confederation": confed_for(name)}
    if isinstance(snap, pd.DataFrame):
        snap = snap.iloc[0]
    return {
        "rating":        round2(snap.get("rating")),
        "rank":          int(snap["rank"]) if not pd.isna(snap.get("rank")) else None,
        "conf_rank":     int(snap["conf_rank"]) if not pd.isna(snap.get("conf_rank")) else None,
        "confederation": clean(snap.get("confederation", "")) or confed_for(name),
    }


# Cumulative counts per (tournament, name, slot) tallied oldest-first so each
# edition reflects the running total THROUGH that edition (matches MESSI).
champions = {}
for tour in MEDAL_TOURNAMENTS:
    years = sorted(y for (t, y) in edition_results if t == tour)
    champ_counts, ru_counts, third_counts = {}, {}, {}
    entries_oldest_first = []
    for year in years:
        res = edition_results[(tour, year)]
        gold, silver, bronze = res.get(1), res.get(2), res.get(3)

        def team_block(name, count_key, counter, _tour=tour, _year=year):
            if not name:
                return None
            counter[name] = counter.get(name, 0) + 1
            info = edition_team_info(name, _tour, _year)
            return {
                "team":          name,
                "flag":          flag_emoji(name),
                "confederation": info["confederation"],
                "rating":        info["rating"],
                "rank":          info["rank"],
                "conf_rank":     info["conf_rank"],
                count_key:       counter[name],
                "wl":            edition_team_wl(_tour, _year, name),
            }

        entries_oldest_first.append({
            "season":     year,
            "host_flags": "",  # host data not modeled for ICHIRO; UI hides empty
            "champion":   team_block(gold,   "title_count",     champ_counts),
            "runner_up":  team_block(silver, "runner_up_count", ru_counts),
            "third":      team_block(bronze, "third_count",      third_counts),
        })

    # Mark pre_rated editions (no medalist has a rated snapshot at the final
    # date) and strip the now-meaningless rating/rank fields. Mirrors MESSI.
    for entry in entries_oldest_first:
        rated = any(
            entry.get(slot) and entry[slot].get("rating") is not None
            for slot in ("champion", "runner_up", "third")
        )
        if rated:
            continue
        entry["pre_rated"] = True
        for slot in ("champion", "runner_up", "third"):
            tb = entry.get(slot)
            if not tb:
                continue
            for k in ("rating", "rank", "conf_rank", "confederation"):
                tb.pop(k, None)

    champions[tour] = list(reversed(entries_oldest_first))  # newest-first

jdump(champions, os.path.join(DATA, "champions.json"))
print(f"  champions.json: {sum(len(v) for v in champions.values())} editions "
      f"across {len(champions)} tournaments")

# Remove the legacy flat Medals output if it exists from a prior run.
_legacy_medals = os.path.join(DATA, "medals.json")
if os.path.exists(_legacy_medals):
    os.remove(_legacy_medals)
    print("  removed legacy medals.json")


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
