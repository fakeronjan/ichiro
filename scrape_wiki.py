# ============================================================
# ICHIRO - International men's baseball scraper
# Parses {{bb res}} result-table rows and {{Linescore}} box scores
# from Wikipedia tournament pages.
#
# Structure: event main page -> per-pool/group sub-pages -> games.
#   - {{bb res|<date>|<time>|{{bb-rt|CODE}}|[[..|A-B]]|'''{{bb|CODE}}'''|...}}
#       road team / score (en-dash) / home team; winner bolded with '''.
#   - {{Linescore |Date=.. |Road={{bb|..}} |RoadAbr=CODE |RR=n .. |Home=.. |HR=n}}
#       detailed box; RR / HR (or RR / RR-home variants) = total runs.
#
# Codes are 3-letter IOC codes. All games are at neutral/host venues, so the
# downstream engine tags them neutral=True (HCA=0).
#
# DATA-INTEGRITY (fleet lesson): fetch_wikitext NEVER raises (returns '' on
# failure) so a flaky scrape degrades to "no new games" rather than crashing.
# The append-only union in build_dataset() treats all_games.csv as the database:
# a short re-scrape can never delete stored games.
# ============================================================
import re
import sys
import time
import os
import requests
import pandas as pd
from datetime import datetime

WIKI_RAW = "https://en.wikipedia.org/w/index.php?title={title}&action=raw"
HEADERS = {"User-Agent": "ichiro-ratings/1.0 (international baseball ratings; contact via github.com/fakeronjan)"}

ALL_GAMES_CSV = "all_games.csv"

# Tournament tier weights (passed through to engine as a column; engine multiplies
# into the WLS observation weight). WBC + Olympics = top global signal; Premier12
# high; Baseball World Cup mid (era-dependent quality). Documented + tunable.
TIER_WEIGHTS = {
    "World Baseball Classic": 1.0,
    "Olympics":               1.0,
    "WBSC Premier12":         0.85,
    "Baseball World Cup":     0.7,
}

# ------------------------------------------------------------
# Event manifest. Each entry: (main_title, tournament_label, season).
# Sub-pages (pools/groups/championship) are auto-discovered from the main
# page; qualifiers/qualification sub-pages are filtered out (lower tier).
# MODERN core first (WBC + Premier12 + Olympics); World Cup is backfill.
# ------------------------------------------------------------
EVENTS = [
    # --- World Baseball Classic ---
    ("2026 World Baseball Classic", "World Baseball Classic", 2026),
    ("2023 World Baseball Classic", "World Baseball Classic", 2023),
    ("2017 World Baseball Classic", "World Baseball Classic", 2017),
    ("2013 World Baseball Classic", "World Baseball Classic", 2013),
    ("2009 World Baseball Classic", "World Baseball Classic", 2009),
    ("2006 World Baseball Classic", "World Baseball Classic", 2006),
    # --- WBSC Premier12 ---
    ("2024 WBSC Premier12", "WBSC Premier12", 2024),
    ("2019 WBSC Premier12", "WBSC Premier12", 2019),
    ("2015 WBSC Premier12", "WBSC Premier12", 2015),
    # --- Olympics (men's) ---
    ("Baseball at the 2020 Summer Olympics", "Olympics", 2021),  # held 2021
    ("Baseball at the 2008 Summer Olympics", "Olympics", 2008),
    ("Baseball at the 2004 Summer Olympics", "Olympics", 2004),
    ("Baseball at the 2000 Summer Olympics", "Olympics", 2000),
    ("Baseball at the 1996 Summer Olympics", "Olympics", 1996),
    ("Baseball at the 1992 Summer Olympics", "Olympics", 1992),
]

# Baseball World Cup historical backfill (lower priority / bonus).
WORLD_CUP_EVENTS = [
    ("2011 Baseball World Cup", "Baseball World Cup", 2011),
    ("2009 Baseball World Cup", "Baseball World Cup", 2009),
    ("2007 Baseball World Cup", "Baseball World Cup", 2007),
    ("2005 Baseball World Cup", "Baseball World Cup", 2005),
    ("2003 Baseball World Cup", "Baseball World Cup", 2003),
    ("2001 Baseball World Cup", "Baseball World Cup", 2001),
]

# Sub-page suffixes we never want to scrape (lower-tier feeders).
_SKIP_SUBPAGE_RE = re.compile(r"qualif", re.IGNORECASE)


def fetch_wikitext(title, max_retries=3, _redirect_depth=0):
    """Fetch raw wikitext for a page title. Returns '' on failure (NEVER raises)
    so a flaky fetch degrades to 'no new games' rather than crashing the run.
    Follows #REDIRECT (up to 3 hops) -- e.g. '... championship game' redirects
    to '... championship', where the gold-medal box actually lives."""
    url = WIKI_RAW.format(title=requests.utils.quote(title.replace(" ", "_")))
    for attempt in range(max_retries):
        try:
            r = requests.get(url, headers=HEADERS, timeout=30)
            if r.status_code == 404:
                return ""
            r.raise_for_status()
            time.sleep(0.3)
            text = r.text
            m = re.match(r"\s*#REDIRECT\s*\[\[([^\]|#]+)", text, re.IGNORECASE)
            if m and _redirect_depth < 3:
                return fetch_wikitext(m.group(1).strip(), max_retries, _redirect_depth + 1)
            return text
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"  [warn] fetch failed for {title!r}: {e}")
                return ""
            time.sleep(2 ** attempt)
    return ""


# ------------------------------------------------------------
# {{bb res}} parsing
# ------------------------------------------------------------
# A bb res row is a single template; fields are positional, pipe-separated:
#   {{bb res|<date>|<time>|<road>|<score>|<home>|<extra>|<venue>|...}}
# road/home each contain {{bb-rt|CODE}} or {{bb|CODE}}; the WINNER is wrapped
# in '''...'''. score is [[..|A–B]] with an en-dash. We split on top-level
# pipes (ignoring pipes inside nested {{ }} and [[ ]]).

_BB_CODE_RE = re.compile(r"\{\{\s*bb(?:-rt|-rb)?\s*\|\s*([^}|]+)", re.IGNORECASE)
_FLAG_CODE_RE = re.compile(r"\{\{\s*#invoke:\s*flag\s*\|\s*bb\s*\|\s*([^}|]+)", re.IGNORECASE)


def _split_top_pipes(s):
    """Split a template body on top-level '|' (ignoring nested {{}}, [[]])."""
    parts, depth_c, depth_b, cur = [], 0, 0, []
    i = 0
    while i < len(s):
        two = s[i:i+2]
        if two == "{{":
            depth_c += 1; cur.append(two); i += 2
        elif two == "}}":
            depth_c -= 1; cur.append(two); i += 2
        elif two == "[[":
            depth_b += 1; cur.append(two); i += 2
        elif two == "]]":
            depth_b -= 1; cur.append(two); i += 2
        elif s[i] == "|" and depth_c == 0 and depth_b == 0:
            parts.append("".join(cur)); cur = []; i += 1
        else:
            cur.append(s[i]); i += 1
    parts.append("".join(cur))
    return parts


# Section-header -> round-label classifier. Lets the engine identify the gold
# final vs the bronze game when both share a closing date (e.g. Premier12 2024).
_HEADER_RE = re.compile(r"^(={2,6})\s*(.*?)\s*\1\s*$", re.MULTILINE)


def _round_for_position(text, pos):
    """Classify the nearest section header preceding byte offset `pos` into a
    round label: 'final' / 'bronze' / 'semifinal' / '' (group/round-robin)."""
    label = ""
    best = -1
    for m in _HEADER_RE.finditer(text):
        if m.start() > pos:
            break
        best = m.start()
        h = m.group(2).lower()
        if "championship final" in h or h.strip() in ("final", "finals", "gold medal game", "gold medal match"):
            label = "final"
        elif "bronze" in h or "third place" in h or "3rd place" in h:
            label = "bronze"
        elif "semifinal" in h or "semi-final" in h:
            label = "semifinal"
        elif "final" in h and "quarterfinal" not in h and "qualif" not in h:
            # generic "...Final" header (e.g. "Championship round - Final")
            label = "final"
        else:
            label = ""
    return label


def _bb_res_blocks(text):
    """Yield (block_body, start_offset) for each {{bb res|...}} block."""
    i = 0
    low = text
    while True:
        m = re.search(r"\{\{\s*bb res\s*\|", low[i:], re.IGNORECASE)
        if not m:
            return
        start = i + m.start()
        # skip the literal "{{bb res start}}" wrapper (no pipe-list payload)
        depth, j = 0, start
        while j < len(text):
            if text[j:j+2] == "{{":
                depth += 1; j += 2
            elif text[j:j+2] == "}}":
                depth -= 1; j += 2
                if depth == 0:
                    yield text[start:j], start
                    break
            else:
                j += 1
        else:
            return
        i = j


def _extract_code(raw):
    """Pull the 3-letter team code out of a {{bb|..}} / {{bb-rt|..}} / flag invoke."""
    m = _BB_CODE_RE.search(raw)
    if m:
        return m.group(1).strip().upper()
    m = _FLAG_CODE_RE.search(raw)
    if m:
        return m.group(1).strip().upper()
    return None


def _parse_score(raw):
    """Parse 'A-B' (any of hyphen / en-dash / em-dash) from a score cell;
    return (a, b) ints. The character class puts hyphen at the start so it
    reads as a literal, not a range operator (otherwise `[–-]` would try
    to be a decreasing range from en-dash to hyphen and raise re.error)."""
    clean = raw.replace("'''", "")
    clean = re.sub(r"\[\[[^\]]*\|", "", clean).replace("[[", "").replace("]]", "")
    m = re.search(r"(\d+)\s*[-–—]\s*(\d+)", clean)
    if not m:
        return None, None
    return int(m.group(1)), int(m.group(2))


def _parse_date(raw, default_year):
    """Parse many date formats. Falls back to default_year when no year present."""
    s = re.sub(r"\[\[[^\]]*\|", "", raw).replace("[[", "").replace("]]", "")
    s = re.sub(r"\{\{[^}]*\}\}", "", s)
    s = s.strip().rstrip(",").strip()
    # strip a leading weekday if any
    s = re.sub(r"^[A-Za-z]+,\s*", "", s)
    fmts_with_year = ["%b %d, %Y", "%B %d, %Y", "%d %B %Y", "%d %b %Y",
                      "%B %d %Y", "%Y-%m-%d"]
    for f in fmts_with_year:
        try:
            return datetime.strptime(s, f).date()
        except ValueError:
            pass
    # No-year forms ("28 July", "July 28") -> attach default_year
    for f in ["%d %B", "%d %b", "%B %d", "%b %d"]:
        try:
            return datetime.strptime(s, f).replace(year=default_year).date()
        except ValueError:
            pass
    return None


def parse_bb_res(wikitext, tournament, season):
    """Yield game dicts from {{bb res}} rows. Winner inferred from '''bold'''
    but score (A=road, B=home) is authoritative for the margin."""
    rows = []
    for block, pos in _bb_res_blocks(wikitext):
        body = block[block.find("|") + 1: block.rfind("}}")]
        parts = _split_top_pipes(body)
        if len(parts) < 4:
            continue
        date_raw, _time, road_raw, score_raw, home_raw = (
            parts[0], parts[1] if len(parts) > 1 else "",
            parts[2] if len(parts) > 2 else "",
            parts[3] if len(parts) > 3 else "",
            parts[4] if len(parts) > 4 else "",
        )
        road = _extract_code(road_raw)
        home = _extract_code(home_raw)
        sa, sb = _parse_score(score_raw)  # A = road runs, B = home runs
        if not (road and home and sa is not None and sb is not None):
            continue
        gdate = _parse_date(date_raw, season)
        if gdate is None:
            continue
        rows.append({
            "date": gdate, "tournament": tournament, "season": season,
            "road_team": road, "road_runs": sa,
            "home_team": home, "home_runs": sb,
            "round": _round_for_position(wikitext, pos),
        })
    return rows


# ------------------------------------------------------------
# {{Linescore}} parsing
# ------------------------------------------------------------
# Named fields. Road/Home hold {{bb|..}} or {{#invoke:flag|bb|..}}; RoadAbr/
# HomeAbr give the code directly (preferred). RR / HR are total runs (RR=road,
# HR or RR-home). Beware RoadHR / HomeHR (home-run hitters) - must NOT match HR.

def _linescore_field(block, key):
    """Extract |Key=value (value up to next top-level |Field= or closing }})."""
    m = re.search(
        rf"\|\s*{key}\s*=\s*(.*?)(?=\n?\s*\|\s*[A-Za-z][A-Za-z0-9]*\s*=|\}}\}})",
        block, re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else ""


def _linescore_blocks(text):
    i = 0
    while True:
        m = re.search(r"\{\{\s*linescore\b", text[i:], re.IGNORECASE)
        if not m:
            return
        start = i + m.start()
        depth, j = 0, start
        while j < len(text):
            if text[j:j+2] == "{{":
                depth += 1; j += 2
            elif text[j:j+2] == "}}":
                depth -= 1; j += 2
                if depth == 0:
                    yield text[start:j], start
                    break
            else:
                j += 1
        else:
            return
        i = j


def parse_linescore(wikitext, tournament, season):
    rows = []
    for block, pos in _linescore_blocks(wikitext):
        road_abr = _linescore_field(block, "RoadAbr")
        home_abr = _linescore_field(block, "HomeAbr")
        road = (road_abr.strip().upper()[:4] if road_abr
                else _extract_code(_linescore_field(block, "Road")))
        home = (home_abr.strip().upper()[:4] if home_abr
                else _extract_code(_linescore_field(block, "Home")))
        rr = _linescore_field(block, "RR")
        # home total runs: usually HR= ; but RoadHR/HomeHR are HR-hitters. The
        # named-field regex anchors on the exact key, so HR= is the home total.
        hr = _linescore_field(block, "HR")
        try:
            road_runs = int(re.search(r"-?\d+", rr).group(0))
            home_runs = int(re.search(r"-?\d+", hr).group(0))
        except (AttributeError, ValueError):
            continue
        if not (road and home):
            continue
        date_raw = _linescore_field(block, "Date")
        gdate = _parse_date(date_raw, season)
        if gdate is None:
            continue
        rows.append({
            "date": gdate, "tournament": tournament, "season": season,
            "road_team": road, "road_runs": road_runs,
            "home_team": home, "home_runs": home_runs,
            "round": _round_for_position(wikitext, pos),
        })
    return rows


# ------------------------------------------------------------
# Sub-page discovery + per-event scrape
# ------------------------------------------------------------
def discover_subpages(main_title, wikitext):
    """Sub-pages that hold games. Two reference styles, both handled:
      1. [[<main_title> <suffix>]] wiki-links ('Pool A' / '– Championship').
      2. {{main|<main_title> <suffix>}} and {{#lst:<main_title> <suffix>|label}}
         transclusions. Modern WBC pool/knockout pages are referenced ONLY this
         way (not as plain wiki-links), so without this they were invisible.
    Qualifier / qualification feeders are filtered out."""
    subs = set()
    base = re.escape(main_title)
    for m in re.finditer(rf"\[\[({base}[^\]|#]+)", wikitext):
        page = m.group(1).strip()
        if not _SKIP_SUBPAGE_RE.search(page):
            subs.add(page)
    for m in re.finditer(rf"\{{\{{\s*(?:main|#lst|#lstx)\s*[:|]\s*({base}[^|}}#\n]+)",
                         wikitext, re.IGNORECASE):
        page = m.group(1).strip()
        if not _SKIP_SUBPAGE_RE.search(page):
            subs.add(page)
    return sorted(subs)


def _baseballbox_blocks(text):
    """Yield each {{Baseballbox ...}} block with balanced braces (case-insensitive)."""
    low = text.lower()
    i = 0
    while True:
        start = low.find("{{baseballbox", i)
        if start == -1:
            return
        depth, j = 0, start
        while j < len(text):
            if text[j:j+2] == "{{":
                depth += 1; j += 2
            elif text[j:j+2] == "}}":
                depth -= 1; j += 2
                if depth == 0:
                    yield text[start:j]
                    break
            else:
                j += 1
        else:
            return
        i = j


def _bbox_field(block, key):
    m = re.search(rf"\|\s*{key}\s*=\s*(.*?)(?=\n\s*\||\}}\}})", block, re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else ""


def parse_baseballboxes(wikitext, tournament, season):
    """Yield game dicts from {{Baseballbox}} blocks. The older Baseball World Cup
    pages (pre-2011) use this named-field template instead of {{bb res}}:
    team1=road, team2=home, score='A &ndash; B'. Winner bolded but the score is
    authoritative for the margin."""
    rows = []
    for block in _baseballbox_blocks(wikitext):
        # Skip pre-tournament exhibition/friendly games (modern WBC pages list
        # dozens of warm-up boxes, often with full-name team templates and
        # non-IOC squads). Only competitive tournament games should count.
        rnd = _bbox_field(block, "round")
        if re.search(r"friendly|exhibition|non-mlb|warm-?up", rnd, re.IGNORECASE):
            continue
        t1 = _extract_code(_bbox_field(block, "team1"))
        t2 = _extract_code(_bbox_field(block, "team2"))
        score = _bbox_field(block, "score").replace("&ndash;", "–").replace("&minus;", "–")
        sa, sb = _parse_score(score)
        gdate = _parse_date(_bbox_field(block, "date"), season)
        if not (t1 and t2 and sa is not None and sb is not None and gdate is not None):
            continue
        rows.append({
            "date": gdate, "tournament": tournament, "season": season,
            "road_team": t1, "road_runs": sa,
            "home_team": t2, "home_runs": sb,
            "round": "",
        })
    return rows


# ------------------------------------------------------------
# LEGACY wikitable parsing (older events with NO box templates)
# ------------------------------------------------------------
# The 1992 & 1996 Olympic baseball pages and the 2007 Baseball World Cup encode
# games as raw wikitables, not {{bb res}}/{{Linescore}}/{{Baseballbox}}:
#   * 1992/1996 (Format A): one R/H/E linescore table per game -- a "Team | 1..9
#     | R | H | E" header with exactly TWO team rows. 1992 uses {{bb|CODE}};
#     1996 uses {{flagicon|Full Country}}. R (total runs) is the 3rd-from-last
#     integer in each team row.
#   * 2007 (Format B): a schedule table -- "Team1 | A-B | Team2 | Time" rows,
#     with "! colspan | <Month Day>" header rows giving the date.

# Full country name -> IOC code (older pages spell teams out via {{flagicon}}).
_NAME_TO_CODE = {
    "United States": "USA", "Cuba": "CUB", "Japan": "JPN", "Nicaragua": "NCA",
    "Netherlands": "NED", "Australia": "AUS", "Italy": "ITA", "South Korea": "KOR",
    "Korea": "KOR", "Chinese Taipei": "TPE", "Taiwan": "TPE", "Spain": "ESP",
    "Dominican Republic": "DOM", "Puerto Rico": "PUR", "Panama": "PAN",
    "South Africa": "RSA", "Mexico": "MEX", "Canada": "CAN", "Venezuela": "VEN",
    "Brazil": "BRA", "China": "CHN", "Colombia": "COL", "Great Britain": "GBR",
    "Israel": "ISR", "Greece": "GRE", "Germany": "GER", "Czech Republic": "CZE",
}

_INT_RE = re.compile(r"-?\d+")
_MONTHS = {m: i for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july",
     "august", "september", "october", "november", "december"], start=1)}


def _team_token_to_code(raw):
    """Resolve a team cell ({{bb|CODE}}, {{flagicon|CODE}}, {{flagicon|Name}},
    or a bold full name) to an IOC code."""
    c = _extract_code(raw)  # handles {{bb|}}/{{bb-rt|}}
    if not c:
        m = re.search(r"\{\{\s*flag(?:icon|country|)?\s*\|\s*([^}|]+)", raw, re.I)
        if m:
            c = m.group(1).strip()
    if not c:
        return None
    c = re.sub(r"'''", "", c).strip()
    if c in _NAME_TO_CODE:
        return _NAME_TO_CODE[c]
    if 2 <= len(c) <= 3 and c.isupper():
        return c
    return _NAME_TO_CODE.get(c)


def _wikitables(text):
    """Yield (table_text, start_offset) for each {| ... |} block (depth-aware)."""
    i = 0
    while True:
        start = text.find("{|", i)
        if start == -1:
            return
        depth, j = 0, start
        while j < len(text) - 1:
            two = text[j:j + 2]
            if two == "{|":
                depth += 1; j += 2
            elif two == "|}":
                depth -= 1; j += 2
                if depth == 0:
                    yield text[start:j], start
                    break
            else:
                j += 1
        else:
            return
        i = j


def _row_runs(row):
    """Total runs (R) for a team row = 3rd-from-last integer (after R come H, E).
    Strips templates / links / bold / table-attribute tokens / hex colors first
    so only the inning + R/H/E numbers remain."""
    s = re.sub(r"\{\{[^{}]*\}\}", " ", row)
    s = re.sub(r"\[\[[^\]]*\]\]", " ", s)
    s = s.replace("'''", " ")
    s = re.sub(r"#[0-9a-fA-F]{3,6}", " ", s)          # css hex colors
    s = re.sub(r'[A-Za-z_][\w-]*\s*=\s*"[^"]*"', " ", s)  # attr="..."
    s = re.sub(r'[A-Za-z_][\w-]*\s*=\s*[\w#%.;:-]+', " ", s)  # align=left, width=6%
    ints = _INT_RE.findall(s)
    return int(ints[-3]) if len(ints) >= 3 else None


def parse_re_tables(wikitext, tournament, season, section_date):
    """Format A: one R/H/E linescore table per game (exactly 2 team rows)."""
    sections = [(m.start(), m.group(2).strip())
                for m in re.finditer(r"^(={2,5})\s*(.*?)\s*\1\s*$", wikitext, re.M)]

    def date_for(pos):
        title = ""
        for s, t in sections:
            if s > pos:
                break
            title = t
        return section_date(title)

    rows = []
    for table, pos in _wikitables(wikitext):
        teams = []
        for rb in re.split(r"\n\|-", table):
            if "{{bb" not in rb and "{{flag" not in rb.lower():
                continue
            code = _team_token_to_code(rb)
            runs = _row_runs(rb)
            if code and runs is not None:
                teams.append((code, runs))
        if len(teams) != 2:
            continue  # not a 2-team game table (e.g. an 8-row standings table)
        gdate = date_for(pos)
        if gdate is None:
            continue
        (ra, sa), (hb, sb) = teams
        rows.append({"date": gdate, "tournament": tournament, "season": season,
                     "road_team": ra, "road_runs": sa,
                     "home_team": hb, "home_runs": sb, "round": ""})
    return rows


def parse_bracket_templates(wikitext, tournament, season):
    """Knockout games encoded in {{Round...}} bracket templates (e.g. the 2007
    World Cup '{{Round8-with third}}'). Each match line is
    '|<date place> | {{flagicon|A}} [[link]] |scoreA| {{flagicon|B}} [[link]] |scoreB'.
    """
    rows = []
    i = 0
    while True:
        start = wikitext.find("{{Round", i)
        if start == -1:
            break
        depth, j = 0, start
        while j < len(wikitext) - 1:
            two = wikitext[j:j + 2]
            if two == "{{":
                depth += 1; j += 2
            elif two == "}}":
                depth -= 1; j += 2
                if depth == 0:
                    break
            else:
                j += 1
        block = wikitext[start:j]
        i = j
        body = re.sub(r"\[\[[^\]]*\]\]", " ", block)  # drop links (they carry pipes)
        for m in re.finditer(
            r"\|\s*([A-Z][a-z]+ \d{1,2})[^|]*\|\s*\{\{\s*flagicon\s*\|\s*([^}|]+)[^}]*\}\}"
            r"\s*\|\s*(\d+)\s*\|\s*\{\{\s*flagicon\s*\|\s*([^}|]+)[^}]*\}\}\s*\|\s*(\d+)",
                body):
            gdate = _parse_date(m.group(1), season)
            a = _team_token_to_code("{{flagicon|%s}}" % m.group(2))
            b = _team_token_to_code("{{flagicon|%s}}" % m.group(4))
            if gdate and a and b and a != b:
                rows.append({"date": gdate, "tournament": tournament, "season": season,
                             "road_team": a, "road_runs": int(m.group(3)),
                             "home_team": b, "home_runs": int(m.group(5)), "round": ""})
    return rows


def parse_schedule_tables(wikitext, tournament, season):
    """Format B: schedule tables -- 'Team1 | A-B | Team2 | Time' rows, dates from
    '! colspan | <Month Day>' header rows."""
    rows = []
    for table, _ in _wikitables(wikitext):
        if "wikitable" not in table:
            continue
        cur_date = None
        for rb in re.split(r"\n\|-", table):
            dm = re.search(r"colspan[^|]*\|\s*([A-Z][a-z]+ \d{1,2})", rb)
            if dm:
                d = _parse_date(dm.group(1), season)
                if d:
                    cur_date = d
                continue
            flags = re.findall(r"\{\{\s*flag(?:icon|country|)?\s*\|\s*([^}|]+)", rb, re.I)
            body = re.sub(r"\{\{[^{}]*\}\}", " ", rb)
            sc = re.search(r"(\d+)\s*[-–—]\s*(\d+)", body)
            if cur_date and len(flags) >= 2 and sc:
                a = _team_token_to_code("{{flagicon|%s}}" % flags[0])
                b = _team_token_to_code("{{flagicon|%s}}" % flags[1])
                if a and b and a != b:
                    rows.append({"date": cur_date, "tournament": tournament,
                                 "season": season, "road_team": a, "road_runs": int(sc.group(1)),
                                 "home_team": b, "home_runs": int(sc.group(2)), "round": ""})
    return rows


# 1992 Barcelona baseball: section -> (month, day), read off the page's own
# Schedule narrative (the round tables carry no dates of their own).
_OLY1992_DATES = {
    "Round 1": (7, 26), "Round 2": (7, 27), "Round 3": (7, 28), "Round 4": (7, 29),
    "Round 5": (7, 31), "Round 6": (8, 1), "Round 7": (8, 2),
    "Semifinals": (8, 4), "Bronze medal match": (8, 5), "Final": (8, 5),
}


def scrape_legacy_event(main_title, tournament, season):
    """Scrape an older event whose games live in raw wikitables (no box
    templates). Picks Format A (R/H/E tables) or Format B (schedule) by content."""
    from datetime import date
    wt = fetch_wikitext(main_title)
    if not wt:
        print(f"  [warn] no wikitext for legacy event {main_title!r}")
        return pd.DataFrame()

    if season == 1992:
        def section_date(title):
            md = _OLY1992_DATES.get(title.strip())
            return date(1992, *md) if md else None
    else:  # 1996 etc.: date is in the section header, e.g. "Day 1 (July 20)"
        def section_date(title):
            m = re.search(r"\(([A-Za-z]+)\s+(\d{1,2})\)", title)
            if m and m.group(1).lower() in _MONTHS:
                return date(season, _MONTHS[m.group(1).lower()], int(m.group(2)))
            return None

    rows = parse_re_tables(wt, tournament, season, section_date) + \
        parse_schedule_tables(wt, tournament, season) + \
        parse_bracket_templates(wt, tournament, season)
    df = pd.DataFrame(rows)
    if len(df):
        df["_gamekey"] = df.apply(
            lambda r: frozenset(((r["road_team"], r["road_runs"]),
                                 (r["home_team"], r["home_runs"]))), axis=1)
        df = df.drop_duplicates(subset="_gamekey", keep="first").drop(columns="_gamekey")
        df["tier"] = TIER_WEIGHTS.get(tournament, 1.0)
        df["neutral"] = True
    return df


def scrape_event(main_title, tournament, season):
    """Scrape one event: main page + auto-discovered pool/championship sub-pages.
    {{bb res}}, {{Linescore}}, and {{Baseballbox}} are all parsed on every page.
    Games are deduped by (date, road, home, road_runs, home_runs)."""
    main_wt = fetch_wikitext(main_title)
    if not main_wt:
        print(f"  [warn] no wikitext for event {main_title!r}")
        return pd.DataFrame()
    pages = [main_title] + discover_subpages(main_title, main_wt)
    all_rows, cache = [], {main_title: main_wt}
    for p in pages:
        wt = cache.get(p) or fetch_wikitext(p)
        rows = (parse_bb_res(wt, tournament, season)
                + parse_linescore(wt, tournament, season)
                + parse_baseballboxes(wt, tournament, season))
        if rows:
            print(f"    {p}: {len(rows)} games")
        all_rows.extend(rows)
    df = pd.DataFrame(all_rows)
    if len(df):
        if "round" not in df.columns:
            df["round"] = ""
        df["round"] = df["round"].fillna("")
        # Identify a game by its unordered {team: runs} mapping WITHIN the
        # edition, NOT by date. Date is the least-reliable field: the same game
        # can be dated differently on different pages (e.g. the 2017 NED-PUR
        # semifinal is "Mar 20" on the main page but "Mar 19" on the
        # championship sub-page -- a Wikipedia typo). Keying on the team/score
        # mapping collapses those cross-source date conflicts AND home/road
        # swaps, while keeping reversed-score rematches (e.g. pool-then-tiebreaker
        # with different scores) distinct. Two genuinely different games with
        # identical teams AND identical score don't occur within one edition.
        # On a residual conflict we prefer a round-labeled row, then the later
        # date (deterministic; the rating impact of a 1-day shift is nil).
        df["_gamekey"] = df.apply(
            lambda r: frozenset(((r["road_team"], r["road_runs"]),
                                 (r["home_team"], r["home_runs"]))), axis=1)
        df["_has_round"] = (df["round"] != "").astype(int)
        df = (df.sort_values(["_has_round", "date"], ascending=[False, False])
                .drop_duplicates(subset="_gamekey", keep="first")
                .drop(columns=["_has_round", "_gamekey"]))
        df["tier"] = TIER_WEIGHTS.get(tournament, 1.0)
        df["neutral"] = True
    return df


# ------------------------------------------------------------
# Append-only union (data-integrity guard, COBI pattern)
# ------------------------------------------------------------
def union_with_existing(fresh_df, path=ALL_GAMES_CSV):
    """Treat the committed CSV as the database. Fresh rows win on conflict (so
    score corrections land); games already stored that this run missed are
    PRESERVED. History can only grow or be corrected, never silently shrink."""
    if not os.path.exists(path):
        return fresh_df
    prev = pd.read_csv(path)
    prev["date"] = pd.to_datetime(prev["date"], errors="coerce").dt.date
    fresh = fresh_df.copy()
    fresh["date"] = pd.to_datetime(fresh["date"], errors="coerce").dt.date
    key = ["date", "road_team", "home_team", "road_runs", "home_runs"]
    f = fresh.copy();  f["_pri"] = 0
    p = prev.copy();   p["_pri"] = 1
    combined = pd.concat([f, p], ignore_index=True, sort=False)
    combined = combined.sort_values("_pri").drop_duplicates(subset=key, keep="first")
    fk = set(map(tuple, fresh[key].astype(str).values))
    preserved = sum(1 for k in map(tuple, prev[key].astype(str).values) if k not in fk)
    if preserved:
        print(f"[db-union] preserved {preserved:,} stored games this run's fetch "
              f"did not return (flaky source -- not deleting history)")
    return combined.drop(columns=["_pri"]).reset_index(drop=True)


# Older events with no box templates (games live in raw wikitables / bracket
# templates) -- routed to scrape_legacy_event instead of scrape_event.
_LEGACY_TITLES = {
    "Baseball at the 1992 Summer Olympics",
    "Baseball at the 1996 Summer Olympics",
    "2007 Baseball World Cup",
}


def build_dataset(events, write=True):
    frames = []
    for main_title, tournament, season in events:
        print(f"== {main_title} ({tournament} {season}) ==")
        df = (scrape_legacy_event(main_title, tournament, season)
              if main_title in _LEGACY_TITLES
              else scrape_event(main_title, tournament, season))
        if len(df):
            print(f"   -> {len(df)} games")
            frames.append(df)
    fresh = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not len(fresh):
        print("No games scraped this run.")
        return fresh
    fresh = fresh.sort_values(["date", "road_team", "home_team"]).reset_index(drop=True)
    merged = union_with_existing(fresh)
    merged = merged.sort_values(["date", "road_team", "home_team"]).reset_index(drop=True)
    if write:
        merged.to_csv(ALL_GAMES_CSV, index=False)
        print(f"\nWrote {ALL_GAMES_CSV}: {len(merged)} games total.")
    return merged


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "modern"
    if mode == "worldcup":
        ev = WORLD_CUP_EVENTS
    elif mode == "all":
        ev = EVENTS + WORLD_CUP_EVENTS
    else:
        ev = EVENTS  # modern core
    df = build_dataset(ev)
    if len(df):
        print(f"\n=== Totals by tournament ===")
        print(df.groupby("tournament").size().to_string())
        teams = sorted(set(df["road_team"]) | set(df["home_team"]))
        print(f"\ndistinct team codes ({len(teams)}): {teams}")
