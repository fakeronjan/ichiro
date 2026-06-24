"""
generate_data.py - reads ichiro_ratings.csv + all_games.csv + tournament_podiums.csv
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
  - margins are "runs" (baseball), no draws - W-L records only
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
import glob
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
    # Netherlands Antilles (AHO) is defunct - no flag emoji.
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
# its rating - not a global present-day window (which would, e.g., show a
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


# ── Last game per (team, date) - the game that produced each snapshot ────────
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
# PODIUMS - per-edition gold/silver/bronze come straight from the podium table.
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
# TOURNAMENT FINAL DATES (per tournament+year) - labels, GOAT + year anchors
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

_CURRENT_YEAR = datetime.now(timezone.utc).year  # in-progress-year gate
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
        chosen = (last_game, "End of year" if year < _CURRENT_YEAR else "Current")
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
# 2) CURRENT STANDINGS (latest snapshot) - compatibility output
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
# 3) GOAT TABLE - top single-snapshot ratings at FLAGSHIP finals
# ============================================================
print("Writing goat_teams.json...")
# Eligibility: medaled (1st/2nd/3rd) in a FLAGSHIP tournament (WBC or Olympics)
# that year. Anchor the rating at THAT tournament's final date. Premier12 +
# Baseball World Cup still feed the rolling rating but do NOT anchor a GOAT
# entry - this dedupes to one entry per team per flagship edition and drops the
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

# ── World Baseball Classic per-edition record (pool + knockout split) ────────
# Mirrors MESSI's WC view. Baseball has no draws, so records are plain W-L:
# top line = POOL play, bottom line = KNOCKOUT. Knockout (single-elimination)
# games come from the 'round' tags (semifinal/final/bronze - covers 2006-2017,
# which had no quarterfinals) plus a Wikipedia-verified set for the editions
# that added an untagged quarterfinal round (2023, 2026). Each match line also
# carries the opponent's pre-match rank/rating. games team names are already
# canonical full names here (mapped at load), matching the ratings `name`.
from bisect import bisect_left, bisect_right

_team_snaps = {}
for _nm, _sub in df[["name", "date", "rank", "rating"]].dropna(subset=["date"]).groupby("name"):
    _sub = _sub.sort_values("date")
    _team_snaps[_nm] = (list(_sub["date"]), list(_sub["rank"]), list(_sub["rating"]))


def country_standing(country, match_date, inclusive=False):
    """A country's (rank, rating) relative to match_date; None if unknown.
    inclusive=False -> latest snapshot strictly BEFORE the date (going-in value);
    inclusive=True  -> latest snapshot ON or before the date (the game-day update,
    i.e. the post-game value once that day's result is baked in)."""
    snap = _team_snaps.get(country)
    if not snap:
        return None
    dates, ranks, ratings = snap
    i = bisect_right(dates, match_date) if inclusive else bisect_left(dates, match_date)
    if i == 0:
        return None
    rk, rt = ranks[i - 1], ratings[i - 1]
    if pd.isna(rk) or pd.isna(rt):
        return None
    return int(rk), round(float(rt), 2)


def opp_standing(opp, match_date):
    """Opponent (rank, rating) as of just before match_date; None if unknown."""
    return country_standing(opp, match_date, inclusive=False)


# Knockout games whose round the scraper left untagged. Verified vs Wikipedia:
# /wiki/2023_World_Baseball_Classic and /wiki/2026_World_Baseball_Classic.
_WBC_KO_EXTRA = {
    ("2023-03-15", frozenset({"Australia", "Cuba"})),          # QF (shares date with pool games)
    ("2023-03-16", frozenset({"Italy", "Japan"})),             # QF
    ("2023-03-17", frozenset({"Puerto Rico", "Mexico"})),      # QF
    ("2023-03-18", frozenset({"United States", "Venezuela"})), # QF
    ("2023-03-21", frozenset({"United States", "Japan"})),     # final (untagged)
}


def _wbc_is_knockout(date_str, home, away, rnd):
    if isinstance(rnd, str) and rnd.strip() in ("semifinal", "final", "bronze"):
        return True
    if (date_str, frozenset({home, away})) in _WBC_KO_EXTRA:
        return True
    # 2026 added quarterfinals; pool play ended Mar 11, knockout ran Mar 13-17.
    return "2026-03-13" <= date_str <= "2026-03-17"


_wbc_team_games = {}
for _, _g in games[games["tournament"] == "World Baseball Classic"].iterrows():
    if pd.isna(_g["date"]) or pd.isna(_g["home_runs"]) or pd.isna(_g["road_runs"]):
        continue
    _yr = int(_g["season"])
    _hr, _rr = int(_g["home_runs"]), int(_g["road_runs"])
    _ds = str(_g["date"])
    _ko = _wbc_is_knockout(_ds, _g["home_team"], _g["road_team"], _g.get("round"))
    _neu = bool(_g.get("neutral"))
    _wbc_team_games.setdefault((_g["home_team"], _yr), []).append(
        {"date": _g["date"], "gf": _hr, "ga": _rr, "opp": _g["road_team"], "home": True, "neutral": _neu, "ko": _ko})
    _wbc_team_games.setdefault((_g["road_team"], _yr), []).append(
        {"date": _g["date"], "gf": _rr, "ga": _hr, "opp": _g["home_team"], "home": False, "neutral": _neu, "ko": _ko})

_wbc_record = {}
for (_team, _yr), _gl in _wbc_team_games.items():
    _gl.sort(key=lambda m: m["date"])
    _rec = {"p_w": 0, "p_l": 0, "k_w": 0, "k_l": 0}
    _matches = []
    for _m in _gl:
        _gf, _ga = _m["gf"], _m["ga"]
        _letter = "W" if _gf > _ga else "L"  # baseball: no ties
        _venue = " vs. (N) " if _m["neutral"] else (" vs. " if _m["home"] else " @ ")
        _st = opp_standing(_m["opp"], _m["date"])
        _matches.append({"s": f"{_letter} {_gf}-{_ga}{_venue}{_m['opp']}",
                         "r": _st[0] if _st else None, "g": _st[1] if _st else None,
                         "d": f"{_m['date'].month:02d}-{_m['date'].day:02d}"})
        _tgt = "k" if _m["ko"] else "p"
        _rec[f"{_tgt}_w" if _gf > _ga else f"{_tgt}_l"] += 1
    _rec["matches"] = _matches
    # The selected team's OWN rank/rating walk across the edition: N+1 boundary
    # standings in date order - index 0 is pre-tournament (strictly before the
    # first game), index i+1 is the post-game standing after game i. Mirrors
    # 'matches' so the frontend can render an offset stairstep beside it; the
    # final entry equals the end-of-tournament headline rating. {r, g} = rank,
    # rating (None,None when no snapshot exists, e.g. a country's WBC debut).
    _walk = []
    _pre = country_standing(_team, _gl[0]["date"], inclusive=False)
    _walk.append({"r": _pre[0], "g": _pre[1]} if _pre else {"r": None, "g": None})
    for _m in _gl:
        _ws = country_standing(_team, _m["date"], inclusive=True)
        _walk.append({"r": _ws[0], "g": _ws[1]} if _ws else {"r": None, "g": None})
    _rec["team_walk"] = _walk
    _wbc_record[(_team, _yr)] = _rec


def wbc_record(name, season):
    if pd.isna(season):
        return None
    return _wbc_record.get((name, int(season)))


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
        rows = [
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
        # Attach the WBC edition record (the row the WBC view filters on).
        # Completed editions: to the 'End of World Baseball Classic' anchor.
        # In-progress (no anchor yet): to the latest snapshot, so it shows the
        # current rating + record-so-far (medal keys off finishes, so none yet).
        _wbc_rec = wbc_record(name, season)
        if _wbc_rec and rows:
            _anchors = [row for row in rows if row["year_anchor_label"] == "End of World Baseball Classic"]
            if _anchors:
                for row in _anchors:
                    row["wbc_record"] = _wbc_rec
            else:
                rows[-1]["wbc_record"] = _wbc_rec
                rows[-1]["wbc_in_progress"] = 1
        seasons[int(season)] = rows

    jdump({"team": name, "flag": fl, "confederation": confed, "seasons": seasons},
          os.path.join(TEAMS, f"{team_slug}.json"))

teams_index.sort(key=lambda x: x["name"])
jdump(teams_index, os.path.join(DATA, "teams_index.json"))
# Prune orphaned team files: when the data source renames an entity (e.g.
# 'China PR' -> 'China') or drops one, the old-slug file lingers - unreachable
# from the UI (not in teams_index) but serving frozen stale data. Remove any
# team file whose slug is no longer in the live index.
_live_team_files = {f"{t['slug']}.json" for t in teams_index}
for _f in glob.glob(os.path.join(TEAMS, "*.json")):
    if os.path.basename(_f) not in _live_team_files:
        os.remove(_f)
        print(f"  Pruned orphaned team file: {os.path.basename(_f)}")
print(f"  teams_index.json + {len(teams_index)} team files")


# ============================================================
# 4b) CHAMPIONS TABLE (per tournament edition) - Tournaments tab
# ============================================================
# Emits the MESSI champions.json contract, grouped by tournament, editions
# newest-first. Each medalist cell carries:
#   - cumulative medal count for that country in that tournament
#   - rating / rank / conf_rank at that edition's FINAL snapshot
#   - W-L within that specific edition (ICHIRO addition vs MESSI)
# Editions where no medalist has a rated snapshot at the final date are marked
# pre_rated (UI renders dashes + † footnote, mirroring MESSI).
print("Writing champions.json...")

# Final-day rating/rank lookup keyed by (name, date_str) - reuse the GOAT index.
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
