"""
ICHIRO - International men's baseball power ratings via WLS Massey solver.

Named after Ichiro Suzuki, the international (Japan -> MLB) icon.

Model stack clones GRIFFEY (MLB) with international adaptations borrowed from
MESSI (international soccer):
  - Homebrew WLS Massey solver (no rankit dependency; copy of griffey._solve_massey)
  - Margin cap to suppress blowouts: MARGIN_CAP = 8 runs (baseball-specific)
  - HCA = 0: WBC / Premier12 / Olympics are at neutral / host venues, not true
    home games. Every game is tagged neutral=True by the scraper.
  - Fixed rolling window in CALENDAR time (see WINDOW_YEARS note below)
  - Linear recency decay across the window
  - Per-tournament TIER WEIGHTS folded into the WLS observation weight (MESSI)
  - Medal/podium tracking per edition (gold / silver / bronze)

Data source: all_games.csv produced by scrape_wiki.py (Wikipedia wikitext).

DATA-INTEGRITY: this engine recomputes the full ratings history every run (no
incremental cache). With only ~140 game-days total the full solve is instant,
so the incremental-cache + positional-ranking_id desync risk that bites the
larger fleet sites simply does not apply here. all_games.csv itself is the
append-only database (guarded in scrape_wiki.union_with_existing).
"""

import os
import numpy as np
import pandas as pd

# =========================================================
# CONFIGURATION
# =========================================================

ALL_GAMES_CSV = "all_games.csv"
RATINGS_CSV   = "ichiro_ratings.csv"
PODIUMS_CSV   = "tournament_podiums.csv"

# ---- Margin transform -----------------------------------------------------
# Cap raw run margin at 8 (baseball-specific; the brief's key value). Captures
# the bulk of international games uncapped; only the worst blowouts (mercy-rule
# routs vs minnows) get trimmed so they don't dominate the regression.
MARGIN_TRANSFORM = "cap"
MARGIN_CAP = 8

# ---- Home-court adjustment ------------------------------------------------
# 0 for international baseball: WBC / Premier12 / Olympics play at neutral or
# host venues, not the participants' home parks. Scraper tags neutral=True.
HOME_COURT_ADJUSTMENT = 0.0

# ---- Rolling window (fixed CALENDAR time) ---------------------------------
# International baseball has NO continuous season; game-days are the wrong unit
# (200 game-days spanned ~23yr = essentially all history, so "current form"
# meant nothing). We use a FIXED CALENDAR window instead, matching CARMELO's
# approach for consistency. Baseball is SPARSER than basketball (WBC/Premier12
# fire only every ~4yr, irregularly), so it gets a LONGER window than CARMELO's
# 4yr — 8yr keeps ~2 WBC + ~2 Premier12 cycles in view per snapshot.
# Tuning knob reviewed with user 2026-05-29; tune vs face-validity.
WINDOW_YEARS = 4
WINDOW_DAYS = int(WINDOW_YEARS * 365.25)

# Linear recency decay over calendar time: weight = 1 - (days_ago / WINDOW_DAYS),
# current day = 1.0, oldest in-window approaches the floor. RECENCY_FLOOR keeps
# old editions contributing (sparse data needs every game).
RECENCY_FLOOR = 0.15

# Eligibility: a team must have played at least this many games inside the
# window to appear in a snapshot. Low, because tournament fields are small.
MIN_GAMES = 3

# =========================================================
# TOURNAMENT CONFIGURATION (tier weights + podium events)
# =========================================================

# WLS observation-weight uplift per tournament. Encodes "this game is a more
# reliable signal of senior-national-team strength". Folded multiplicatively
# into the recency weight at solve time. Documented + tunable.
#   WBC + Olympics = top global signal (1.0)
#   Premier12      = high (0.85)
#   Baseball World Cup = mid, era-dependent quality (0.7)
TIER_WEIGHTS = {
    "World Baseball Classic": 1.0,
    "Olympics":               1.0,
    "WBSC Premier12":         0.85,
    "Baseball World Cup":     0.7,
}

# Every modern event here has a true single final (gold) + bronze game.
PODIUM_TOURNAMENTS = {
    "World Baseball Classic", "Olympics", "WBSC Premier12", "Baseball World Cup",
}

# =========================================================
# 3-letter code -> country. IOC codes plus the few Wikipedia variants the
# scraper surfaced (US -> United States, PRI -> Puerto Rico's IOC PUR, etc.).
# Relocation policy is moot here (national teams don't move).
# =========================================================
CODE_TO_COUNTRY = {
    "AUS": "Australia",
    "BRA": "Brazil",
    "CAN": "Canada",
    "CHN": "China",
    "COL": "Colombia",
    "CUB": "Cuba",
    "CZE": "Czech Republic",
    "DOM": "Dominican Republic",
    "ESP": "Spain",
    "GBR": "Great Britain",
    "GRE": "Greece",
    "ISR": "Israel",
    "ITA": "Italy",
    "JPN": "Japan",
    "KOR": "South Korea",
    "MEX": "Mexico",
    "NCA": "Nicaragua",
    "NED": "Netherlands",
    "NCL": "Netherlands",   # rare variant
    "PAN": "Panama",
    "PUR": "Puerto Rico",
    "PRI": "Puerto Rico",   # Premier12 variant of PUR
    "RSA": "South Africa",
    "TPE": "Chinese Taipei",
    "USA": "United States",
    "US":  "United States", # {{bb-rt|US|1960}} variant in 2009 WBC
    "VEN": "Venezuela",
}

# Continental confederation (WBSC-style), for UI grouping / pills.
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


def code_to_country(code):
    code = str(code).strip().upper()
    return CODE_TO_COUNTRY.get(code, code)


# =========================================================
# WLS MASSEY SOLVER (clone of griffey._solve_massey, single zero-sum anchor)
# =========================================================

def _apply_margin_transform(margin, transform, cap):
    m = np.asarray(margin, dtype=float)
    if transform == "raw":
        return m
    if transform == "cap":
        return np.clip(m, -cap, cap)
    raise ValueError(f"Unknown MARGIN_TRANSFORM: {transform}")


def _connected_components(teams, edges):
    """Union-find over the team-game graph (copied from griffey)."""
    parent = {t: t for t in teams}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for h, a in edges:
        union(h, a)
    roots, out = {}, {}
    for t in teams:
        r = find(t)
        if r not in roots:
            roots[r] = len(roots)
        out[t] = roots[r]
    return out


def _solve_massey(window_df):
    """WLS Massey solve on one rolling window. One zero-sum anchor per connected
    component (international baseball is a single confederation-spanning network,
    so usually one component; the per-component anchor is defensive for any
    edition that doesn't connect to the rest).

    window_df needs: home_team, road_team, home_runs, road_runs, weight.
    HCA + margin cap applied here. Returns DataFrame: name, rating, rank, component.
    """
    teams = sorted(set(window_df["home_team"]) | set(window_df["road_team"]))
    team_idx = {t: i for i, t in enumerate(teams)}
    n_teams = len(teams)
    n_games = len(window_df)

    home_runs = window_df["home_runs"].to_numpy(dtype=float)
    road_runs = window_df["road_runs"].to_numpy(dtype=float)
    weights   = window_df["weight"].to_numpy(dtype=float)
    home_names = window_df["home_team"].to_numpy()
    road_names = window_df["road_team"].to_numpy()

    comp_map = _connected_components(teams, zip(home_names, road_names))
    anchor_groups = {}
    for t in teams:
        anchor_groups.setdefault(comp_map[t], []).append(t)
    anchor_keys = sorted(anchor_groups.keys())

    n_rows = n_games + len(anchor_keys)
    X = np.zeros((n_rows, n_teams))
    y = np.zeros(n_rows)
    w = np.zeros(n_rows)

    # margin from HOME perspective: home - road - hca (hca=0 here), then cap.
    raw_margin = home_runs - road_runs - HOME_COURT_ADJUSTMENT
    transformed = _apply_margin_transform(raw_margin, MARGIN_TRANSFORM, MARGIN_CAP)

    for i in range(n_games):
        X[i, team_idx[home_names[i]]] = 1.0
        X[i, team_idx[road_names[i]]] = -1.0
    y[:n_games] = transformed
    w[:n_games] = weights

    for k, key in enumerate(anchor_keys):
        row = n_games + k
        for t in anchor_groups[key]:
            X[row, team_idx[t]] = 1.0
        y[row] = 0.0
        w[row] = 1.0e8

    sqrt_w = np.sqrt(w)
    Xw = X * sqrt_w[:, None]
    yw = y * sqrt_w
    r, *_ = np.linalg.lstsq(Xw, yw, rcond=None)

    out = pd.DataFrame({"name": teams, "rating": r,
                        "component": [comp_map[t] for t in teams]})
    out["rank"] = out["rating"].rank(ascending=False, method="min").astype(int)
    return out


# =========================================================
# DATA PREP
# =========================================================

def prepare_games():
    df = pd.read_csv(ALL_GAMES_CSV)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).copy()

    # Codes -> canonical countries (normalises US/USA, PRI/PUR, etc.)
    df["home_team"] = df["home_team"].map(code_to_country)
    df["road_team"] = df["road_team"].map(code_to_country)

    df["home_runs"] = pd.to_numeric(df["home_runs"], errors="coerce")
    df["road_runs"] = pd.to_numeric(df["road_runs"], errors="coerce")
    df = df.dropna(subset=["home_runs", "road_runs"]).copy()
    df["home_runs"] = df["home_runs"].astype(int)
    df["road_runs"] = df["road_runs"].astype(int)

    # Drop ties (baseball has no ties; a 0-margin row is a data error)
    df = df[df["home_runs"] != df["road_runs"]].copy()

    if "tier" not in df.columns:
        df["tier"] = df["tournament"].map(TIER_WEIGHTS).fillna(1.0)
    df["tier"] = pd.to_numeric(df["tier"], errors="coerce").fillna(1.0)

    df = df.sort_values("date").reset_index(drop=True)
    df["grouped_date_id"] = df.groupby("date").ngroup() + 1
    df["unique_game_id"]  = np.arange(1, len(df) + 1)

    # Result strings for last-game display (home + road perspective)
    df["home_wl"] = np.where(df["home_runs"] > df["road_runs"], "W", "L")
    df["road_wl"] = np.where(df["road_runs"] > df["home_runs"], "W", "L")
    df["home_last_match"] = (
        df["home_wl"] + " " + df["home_runs"].astype(str) + "-" + df["road_runs"].astype(str)
        + " vs. (N) " + df["road_team"] + " (" + df["tournament"] + ")"
    )
    df["road_last_match"] = (
        df["road_wl"] + " " + df["road_runs"].astype(str) + "-" + df["home_runs"].astype(str)
        + " vs. (N) " + df["home_team"] + " (" + df["tournament"] + ")"
    )
    return df


# =========================================================
# RATING LOOP (full recompute; sparse data makes this trivial)
# =========================================================

def compute_ratings(df):
    max_id = int(df["grouped_date_id"].max())
    frames = []
    rid_to_season = (df.drop_duplicates("grouped_date_id")
                       .set_index("grouped_date_id")["season"].to_dict())

    for i in range(1, max_id + 1):
        snap_date = df.loc[df["grouped_date_id"] == i, "date"].max()
        if pd.isnull(snap_date):
            continue
        # Calendar-time window: games within the last WINDOW_DAYS calendar days,
        # so the window means the same thing regardless of schedule sparsity.
        cutoff = snap_date - pd.Timedelta(days=WINDOW_DAYS)
        window = df[(df["date"] <= snap_date) & (df["date"] > cutoff)].copy()
        if not len(window):
            continue
        # Linear recency decay over calendar time, floored, x tournament tier.
        window["days_ago"] = (snap_date - window["date"]).dt.days
        decay = (1.0 - window["days_ago"] / WINDOW_DAYS).clip(lower=RECENCY_FLOOR)
        window["weight"] = decay * window["tier"]

        current_date = snap_date
        season = int(rid_to_season.get(i, current_date.year))

        try:
            ranked = _solve_massey(window)
        except Exception as e:
            print(f"  [skip] grouped_date_id {i} ({current_date.date()}): {e}")
            continue

        # games played in window (eligibility)
        gp = pd.concat([window["home_team"], window["road_team"]]).value_counts()
        ranked["games_played"] = ranked["name"].map(gp).fillna(0).astype(int)

        ranked["ranking_id"]   = i
        ranked["ranking_date"] = current_date.date()
        ranked["season"]       = season
        frames.append(ranked)

    ratings = pd.concat(frames, ignore_index=True)
    ratings = ratings[ratings["games_played"] >= MIN_GAMES].copy()
    ratings.sort_values(["ranking_id", "rank"], inplace=True)
    # re-rank within each snapshot after the eligibility filter
    ratings["rank"] = ratings.groupby("ranking_id")["rating"].rank(
        ascending=False, method="min").astype(int)
    ratings.to_csv(RATINGS_CSV, index=False)
    print(f"{RATINGS_CSV} saved ({len(ratings):,} rows, "
          f"{ratings['ranking_id'].nunique()} snapshots)")
    return ratings


# =========================================================
# PODIUMS (gold / silver / bronze per edition)
# =========================================================

def _won_prev_game(g, team, before_date):
    """Did `team` WIN its most recent game before before_date? (semifinal check)
    Returns True / False / None (no prior game). Used to tell the gold final
    apart from the bronze game when both fall on the same closing date: in the
    final both teams won their semifinal; in the bronze game both lost theirs."""
    prev = g[(g["date"] < before_date) &
             ((g["home_team"] == team) | (g["road_team"] == team))].sort_values("date")
    if prev.empty:
        return None
    last = prev.iloc[-1]
    if last["home_team"] == team:
        return last["home_runs"] > last["road_runs"]
    return last["road_runs"] > last["home_runs"]


def _winner_loser(row):
    if row["home_runs"] > row["road_runs"]:
        return row["home_team"], row["road_team"]
    return row["road_team"], row["home_team"]


def compute_podiums(df):
    """Per edition: gold/silver from the FINAL, bronze from the bronze game.

    The gold final and bronze game often share the closing date. We disambiguate
    with a semifinal bracket-walk (MESSI pattern): the final is the closing game
    whose BOTH participants won their previous game; the bronze game is the one
    whose participants both lost. If the walk is inconclusive (round-robin
    editions with no semis), fall back to the last game of the edition as final.
    """
    has_round = "round" in df.columns
    records = []
    for (tournament, season), g in df.groupby(["tournament", "season"]):
        if tournament not in PODIUM_TOURNAMENTS:
            continue
        g = g.sort_values("date")
        last_date = g["date"].max()
        final_day = g[g["date"] == last_date]
        if not len(final_day):
            continue

        # Preferred signal: a game scraped under a "Championship final" header.
        final = None
        if has_round:
            tagged = g[g["round"] == "final"]
            if len(tagged):
                final = tagged.iloc[-1]

        # Else identify the final among closing-date games via bracket-walk.
        if final is None:
            for _, cand in list(final_day.iterrows())[::-1]:
                h_won = _won_prev_game(g, cand["home_team"], last_date)
                r_won = _won_prev_game(g, cand["road_team"], last_date)
                if h_won and r_won:
                    final = cand
                    break
        if final is None:
            final = final_day.iloc[-1]  # round-robin / inconclusive fallback

        gold, silver = _winner_loser(final)
        finalists = {gold, silver}
        records.append({"tournament": tournament, "season": season, "team": gold,   "finish": 1})
        records.append({"tournament": tournament, "season": season, "team": silver, "finish": 2})

        # Bronze: prefer a "bronze"-tagged game; else closing-date (or day-before)
        # game between two non-finalists.
        bronze_row = None
        if has_round:
            tagged_b = g[g["round"] == "bronze"]
            if len(tagged_b):
                bronze_row = tagged_b.iloc[-1]
        if bronze_row is None:
            cand_days = sorted(g["date"].unique())[-2:]
            bronze_games = g[(g["date"].isin(cand_days)) &
                             (~g["home_team"].isin(finalists)) &
                             (~g["road_team"].isin(finalists))]
            if len(bronze_games):
                bronze_row = bronze_games.iloc[-1]
        if bronze_row is not None:
            bronze, _ = _winner_loser(bronze_row)
            records.append({"tournament": tournament, "season": season, "team": bronze, "finish": 3})

    podiums = pd.DataFrame(records)
    # tournament_podiums.csv is now a CURATED, hand-verified file (authoritative
    # vs Wikipedia). The bracket-walk above is UNRELIABLE: it cannot find 3rd
    # place where no bronze game is played (the WBC ranks its two semifinal
    # losers WITHOUT a game), and it mis-identifies the "final" in round-robin
    # World Cups (got 2001/2005 champions wrong). So it must NEVER overwrite the
    # curated source generate_data.py reads -- write to a debug file only. Same
    # lesson as CARMELO's curated_podiums.csv (bracket-walk podiums abandoned).
    podiums.to_csv("_podiums_autoderived_debug.csv", index=False)
    print(f"[diagnostic] auto-derived podiums -> _podiums_autoderived_debug.csv "
          f"({len(podiums)} recs; NOT used downstream, curated "
          f"tournament_podiums.csv is authoritative)")
    return podiums


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":
    games = prepare_games()
    print(f"Loaded {len(games):,} games, {games['date'].min().date()} .. "
          f"{games['date'].max().date()} ({games['grouped_date_id'].max()} game-days).")
    print(f"Mean home margin = {(games['home_runs'] - games['road_runs']).mean():+.3f} runs "
          f"(neutral venues -> expect ~0).")

    ratings = compute_ratings(games)
    podiums = compute_podiums(games)

    latest_id = ratings["ranking_id"].max()
    latest = ratings[ratings["ranking_id"] == latest_id].sort_values("rank")
    latest_date = latest["ranking_date"].iloc[0]
    print(f"\n=== Top 12 at latest snapshot ({latest_date}) ===")
    print(latest.head(12)[["rank", "name", "rating", "games_played"]].to_string(index=False))

    print("\n=== Auto-derived podium golds (DIAGNOSTIC ONLY -- curated "
          "tournament_podiums.csv is authoritative) ===")
    gold = podiums[podiums["finish"] == 1].sort_values(["tournament", "season"])
    print(gold[["tournament", "season", "team"]].to_string(index=False))
