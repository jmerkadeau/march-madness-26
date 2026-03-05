"""
╔══════════════════════════════════════════════════════════════════════════════╗
║          🏀 MARCH MADNESS ELO BRACKET PREDICTOR — ALL 3 MODELS             ║
║                                                                              ║
║  V1 — Basic ELO      : wins/losses only                                     ║
║  V2 — Enhanced ELO   : + margin of victory, home court, recency weighting  ║
║  V3 — Kitchen Sink   : + conference strength, seed prior blending           ║
║                                                                              ║
║  RATING APPROACH:                                                            ║
║    - 2010-2024: historical calibration (shapes K, reversion behavior)       ║
║    - 2025 only: final ratings reset + clean pass on current season          ║
║    - Aggressive reversion (0.4) between seasons accounts for roster         ║
║      turnover — UConn's 2023-24 dominance won't pollute 2025 ratings       ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1: Imports & Global Config
# ══════════════════════════════════════════════════════════════════════════════

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from collections import defaultdict
import warnings, os, random

warnings.filterwarnings("ignore")

# ── Season Config ─────────────────────────────────────────────────────────────
PREDICT_YEAR      = 2025
CALIBRATION_START = 2010   # Historical seasons used to shape model behavior
CALIBRATION_END   = 2024   # Last historical season (inclusive)
CURRENT_SEASON    = 2025   # Season used for final ratings

# ── ELO Tuning Parameters ─────────────────────────────────────────────────────
STARTING_ELO       = 1500
K_FACTOR           = 20      # V1 base K
K_FACTOR_ENHANCED  = 24      # V2/V3 base K
HOME_ADVANTAGE     = 100
REVERSION_FACTOR   = 0.40    # Aggressive reversion for college basketball roster turnover
                              # (was 0.75 — lowered to discount prior season dominance)
RECENCY_HALF_LIFE  = 30      # Days within a season
SEED_PRIOR_WEIGHT  = 0.15    # V3: blend toward seed-implied ELO
CONF_STRENGTH_W    = 0.10    # V3: conference adjustment scaling
N_SIMULATIONS      = 10_000

# Seed → implied ELO prior
SEED_ELO_MAP = {
    1: 2050, 2: 1950, 3: 1880, 4: 1820, 5: 1770, 6: 1730, 7: 1700, 8: 1670,
    9: 1640, 10: 1620, 11: 1600, 12: 1580, 13: 1540, 14: 1510, 15: 1480, 16: 1450,
}

CONF_STRENGTH = {
    "sec": 90, "big_twelve": 85, "big_ten": 75, "big_east": 80, "acc": 70,
    "pac_twelve": 50, "american": 30, "mountain_west": 25,
    "atlantic_ten": 15, "wcc": 20,
}


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2: Data Loading
# ══════════════════════════════════════════════════════════════════════════════

def load_data():
    DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
    demo_mode = not os.path.exists(os.path.join(DATA_DIR, "MTeams.csv"))

    if demo_mode:
        print("⚠️  Kaggle data not found — running DEMO MODE with synthetic data")
        print("   Get real data: https://www.kaggle.com/competitions/march-machine-learning-mania-2025/data\n")
        return _generate_demo_data()

    print("✅ Kaggle data found — loading real CSVs")
    teams_df   = pd.read_csv(os.path.join(DATA_DIR, "MTeams.csv"))
    reg_df     = pd.read_csv(os.path.join(DATA_DIR, "MRegularSeasonDetailedResults.csv"))
    tourney_df = pd.read_csv(os.path.join(DATA_DIR, "MNCAATourneyDetailedResults.csv"))
    seeds_df   = pd.read_csv(os.path.join(DATA_DIR, "MNCAATourneySeeds.csv"))
    conf_path  = os.path.join(DATA_DIR, "MTeamConferences.csv")
    conf_df    = pd.read_csv(conf_path) if os.path.exists(conf_path) else None
    print(f"   Teams: {len(teams_df)} | Reg games: {len(reg_df)} | Tourney games: {len(tourney_df)}\n")
    return teams_df, reg_df, tourney_df, seeds_df, conf_df


def _generate_demo_data():
    np.random.seed(42); random.seed(42)
    N = 68
    TEAM_IDS = list(range(1101, 1101 + N))
    TEAM_NAMES = [
        "Auburn","Duke","Houston","Iowa St","Tennessee","Alabama","Florida","Michigan St",
        "Texas Tech","Wisconsin","St Johns","Purdue","Arizona","BYU","Missouri","Marquette",
        "Memphis","Ole Miss","Creighton","Mississippi St","UCLA","Gonzaga","Kansas","Baylor",
        "Kentucky","Arkansas","Indiana","Virginia","UConn","Michigan","Ohio St","Illinois",
        "NC State","Xavier","Utah St","VCU","Liberty","Drake","McNeese","High Point",
        "Lipscomb","Akron","Bryant","Stetson","St Francis PA","Texas AM CC","UNCW","Montana",
        "South Dakota","Norfolk St","SIU-Edwardsville","Wofford","Presbyterian","Florida Atlantic",
        "Colorado St","Dayton","Nebraska","Clemson","New Mexico","Louisville","Oregon","Pittsburgh",
        "Wake Forest","Cincinnati","George Mason","Butler","Loyola Chicago","Northwestern",
    ]
    strengths = {tid: np.random.normal(1500, 150) for tid in TEAM_IDS}
    teams_df  = pd.DataFrame({"TeamID": TEAM_IDS, "TeamName": TEAM_NAMES})

    def make_game(a, b, season, day, neutral=False):
        sa, sb = strengths[a], strengths[b]
        bonus  = 0 if neutral else HOME_ADVANTAGE * 0.5
        prob_a = 1 / (1 + 10 ** ((sb - sa - bonus) / 400))
        ws = int(np.random.normal(70 + (sa - 1500) / 30, 8))
        ls = int(np.random.normal(70 + (sb - 1500) / 30, 8))
        winner = a if random.random() < prob_a else b
        loser  = b if winner == a else a
        ws, ls = (ws, ls) if ws > ls else (ls, ws)
        return {"Season": season, "DayNum": day, "WTeamID": winner, "LTeamID": loser,
                "WScore": ws, "LScore": ls, "WLoc": "N" if neutral else "H"}

    reg, trn = [], []
    for season in range(2010, 2026):
        for _ in range(N * 15):
            a, b = random.sample(TEAM_IDS, 2)
            reg.append(make_game(a, b, season, random.randint(30, 132)))
        if season < 2025:
            for _ in range(67):
                a, b = random.sample(TEAM_IDS, 2)
                trn.append(make_game(a, b, season, 136 + random.randint(0, 18), neutral=True))

    regions = ["E", "W", "S", "M"]
    shuffled = random.sample(TEAM_IDS, 64)
    seeds, idx = [], 0
    for r in regions:
        for s in range(1, 17):
            seeds.append({"Season": PREDICT_YEAR, "Seed": f"{r}{s:02d}", "TeamID": shuffled[idx]})
            idx += 1

    confs = list(CONF_STRENGTH.keys())
    conf_df = pd.DataFrame({
        "Season": PREDICT_YEAR, "TeamID": TEAM_IDS,
        "ConfAbbrev": [random.choice(confs) for _ in TEAM_IDS],
    })
    return teams_df, pd.DataFrame(reg), pd.DataFrame(trn), pd.DataFrame(seeds), conf_df


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3: Shared ELO Math Utilities
# ══════════════════════════════════════════════════════════════════════════════

def expected_score(elo_a, elo_b):
    return 1 / (1 + 10 ** ((elo_b - elo_a) / 400))

def mov_multiplier(score_diff, elo_diff):
    return np.log(abs(score_diff) + 1) * (2.2 / (elo_diff * 0.001 + 2.2))

def recency_weight(day_num, max_day=132, half_life=RECENCY_HALF_LIFE):
    return 2 ** ((day_num - max_day) / half_life)

def revert_to_mean(elo_dict, factor=REVERSION_FACTOR, mean=STARTING_ELO):
    return {tid: mean + factor * (elo - mean) for tid, elo in elo_dict.items()}

def games_for_seasons(reg_df, tourney_df, seasons):
    cols = ["Season", "DayNum", "WTeamID", "LTeamID", "WScore", "LScore", "WLoc"]
    reg = reg_df[reg_df["Season"].isin(seasons)][cols].copy()
    trn = tourney_df[tourney_df["Season"].isin(seasons)][cols].copy()
    return pd.concat([reg, trn]).sort_values(["Season", "DayNum"]).reset_index(drop=True)

def games_for_season_reg_only(reg_df, season):
    """Current season: regular season only (no tourney results yet)."""
    cols = ["Season", "DayNum", "WTeamID", "LTeamID", "WScore", "LScore", "WLoc"]
    return reg_df[reg_df["Season"] == season][cols].sort_values("DayNum").reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4: Two-Phase Training
#
#   Phase 1 — Calibration (2010-2024):
#     Runs full ELO updates across historical seasons with season-end reversion.
#     This teaches the model the right K/reversion behavior but we DON'T use
#     these ratings directly — they're just a warm start.
#
#   Phase 2 — Current Season (2025):
#     After calibration, HARD RESET all teams to mean (reversion=0), then run
#     a clean pass on 2025 regular season only. This ensures final ratings
#     reflect only this year's performance, not UConn's 2023-24 dynasty.
# ══════════════════════════════════════════════════════════════════════════════

def _run_elo_pass(games, elos, update_fn):
    """Generic ELO pass with automatic season-boundary reversion."""
    cur_season = None
    for _, g in games.iterrows():
        if g["Season"] != cur_season:
            if cur_season is not None:
                reverted = revert_to_mean(dict(elos))
                elos.clear()
                elos.update(reverted)
            cur_season = g["Season"]
        w_delta, l_delta = update_fn(g, elos)
        elos[int(g["WTeamID"])] += w_delta
        elos[int(g["LTeamID"])] += l_delta
    return elos


def _hard_reset(elos):
    """Full reversion to mean — wipes historical carry-over before current season pass."""
    return defaultdict(lambda: STARTING_ELO,
                       {tid: STARTING_ELO for tid in elos})


# ── Update rules ──────────────────────────────────────────────────────────────

def _basic_update(g, elos, k=K_FACTOR):
    w, l = int(g["WTeamID"]), int(g["LTeamID"])
    ew = expected_score(elos[w], elos[l])
    return k * (1 - ew), k * (0 - (1 - ew))


def _enhanced_update(g, elos, k=K_FACTOR_ENHANCED):
    w, l = int(g["WTeamID"]), int(g["LTeamID"])
    loc  = g.get("WLoc", "N")
    home = HOME_ADVANTAGE if loc == "H" else (-HOME_ADVANTAGE if loc == "A" else 0)
    ew   = expected_score(elos[w] + home, elos[l])
    mult = mov_multiplier(g["WScore"] - g["LScore"], elos[w] - elos[l])
    rw   = recency_weight(g["DayNum"])
    upd  = k * mult * rw * (1 - ew)
    return upd, -upd


def _make_kitchen_sink_update(conf_adj):
    def _update(g, elos, k=K_FACTOR_ENHANCED):
        w, l = int(g["WTeamID"]), int(g["LTeamID"])
        loc  = g.get("WLoc", "N")
        wc   = conf_adj.get(w, 0) * CONF_STRENGTH_W
        lc   = conf_adj.get(l, 0) * CONF_STRENGTH_W
        home = HOME_ADVANTAGE if loc == "H" else (-HOME_ADVANTAGE if loc == "A" else 0)
        ew   = expected_score(elos[w] + home + wc, elos[l] + lc)
        mult = mov_multiplier(g["WScore"] - g["LScore"], elos[w] - elos[l])
        rw   = recency_weight(g["DayNum"])
        upd  = k * mult * rw * (1 - ew)
        return upd, -upd
    return _update


def build_conf_adjustments(conf_df, season):
    if conf_df is None:
        return {}
    sc = conf_df[conf_df["Season"] == season]
    result = {}
    for _, row in sc.iterrows():
        abbrev = str(row["ConfAbbrev"]).lower().replace("-", "_").replace(" ", "_")
        strength = next((v for k, v in CONF_STRENGTH.items() if k in abbrev or abbrev in k), 0)
        result[int(row["TeamID"])] = strength
    return result


def blend_seed_prior(elos, seeds_df, season, weight=SEED_PRIOR_WEIGHT):
    s = seeds_df[seeds_df["Season"] == season]
    blended = dict(elos)
    for _, row in s.iterrows():
        tid = int(row["TeamID"])
        try:
            seed_num = int("".join(filter(str.isdigit, str(row["Seed"])[:3])))
        except Exception:
            seed_num = 8
        seed_elo = SEED_ELO_MAP.get(seed_num, STARTING_ELO)
        current  = blended.get(tid, STARTING_ELO)
        blended[tid] = (1 - weight) * current + weight * seed_elo
    return blended


# ── Two-phase training functions ──────────────────────────────────────────────

def train_basic_elo(reg_df, tourney_df):
    """V1 two-phase: calibrate on 2010-2024, hard reset, run 2025 only."""
    cal_seasons = list(range(CALIBRATION_START, CALIBRATION_END + 1))
    elos = defaultdict(lambda: STARTING_ELO)

    # Phase 1: calibration
    cal_games = games_for_seasons(reg_df, tourney_df, cal_seasons)
    elos = _run_elo_pass(cal_games, elos, _basic_update)

    # Phase 2: hard reset → 2025 regular season only
    elos = _hard_reset(elos)
    cur_games = games_for_season_reg_only(reg_df, CURRENT_SEASON)
    for _, g in cur_games.iterrows():
        w_delta, l_delta = _basic_update(g, elos)
        elos[int(g["WTeamID"])] += w_delta
        elos[int(g["LTeamID"])] += l_delta

    return dict(elos)


def train_enhanced_elo(reg_df, tourney_df):
    """V2 two-phase: calibrate on 2010-2024, hard reset, run 2025 only."""
    cal_seasons = list(range(CALIBRATION_START, CALIBRATION_END + 1))
    elos = defaultdict(lambda: STARTING_ELO)

    # Phase 1: calibration
    cal_games = games_for_seasons(reg_df, tourney_df, cal_seasons)
    elos = _run_elo_pass(cal_games, elos, _enhanced_update)

    # Phase 2: hard reset → 2025 regular season only
    elos = _hard_reset(elos)
    cur_games = games_for_season_reg_only(reg_df, CURRENT_SEASON)
    for _, g in cur_games.iterrows():
        w_delta, l_delta = _enhanced_update(g, elos)
        elos[int(g["WTeamID"])] += w_delta
        elos[int(g["LTeamID"])] += l_delta

    return dict(elos)


def train_kitchen_sink_elo(reg_df, tourney_df, conf_df):
    """V3 two-phase: calibrate on 2010-2024, hard reset, run 2025 with conf + seed blend."""
    cal_seasons = list(range(CALIBRATION_START, CALIBRATION_END + 1))
    elos = defaultdict(lambda: STARTING_ELO)

    # Phase 1: calibration with per-season conference adjustments
    cal_games = games_for_seasons(reg_df, tourney_df, cal_seasons)
    cur_season_cal, conf_adj = None, {}
    for _, g in cal_games.iterrows():
        if g["Season"] != cur_season_cal:
            if cur_season_cal is not None:
                reverted = revert_to_mean(dict(elos))
                elos.clear(); elos.update(reverted)
            cur_season_cal = g["Season"]
            conf_adj = build_conf_adjustments(conf_df, int(g["Season"]))
        update_fn = _make_kitchen_sink_update(conf_adj)
        w_delta, l_delta = update_fn(g, elos)
        elos[int(g["WTeamID"])] += w_delta
        elos[int(g["LTeamID"])] += l_delta

    # Phase 2: hard reset → 2025 regular season only
    elos = _hard_reset(elos)
    conf_adj_2025 = build_conf_adjustments(conf_df, CURRENT_SEASON)
    update_fn_2025 = _make_kitchen_sink_update(conf_adj_2025)
    cur_games = games_for_season_reg_only(reg_df, CURRENT_SEASON)
    for _, g in cur_games.iterrows():
        w_delta, l_delta = update_fn_2025(g, elos)
        elos[int(g["WTeamID"])] += w_delta
        elos[int(g["LTeamID"])] += l_delta

    return dict(elos)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5: Bracket Simulation Engine
# ══════════════════════════════════════════════════════════════════════════════

SEED_ORDER  = [1, 16, 8, 9, 5, 12, 4, 13, 6, 11, 3, 14, 7, 10, 2, 15]
REGION_FULL = {"E": "East", "W": "West", "S": "South", "M": "Midwest"}


def parse_bracket(seeds_df, season):
    s = seeds_df[seeds_df["Season"] == season].copy()
    s["region"]   = s["Seed"].str[0]
    s["seed_num"] = s["Seed"].str[1:3].str.replace("a", "").str.replace("b", "").astype(int)
    s["TeamID"]   = s["TeamID"].astype(int)
    bracket = {}
    for region, grp in s.groupby("region"):
        bracket[region] = sorted(zip(grp["seed_num"], grp["TeamID"]))
    return bracket


def first_round_pairs(region_teams):
    seed_to_team = {s: t for s, t in region_teams}
    return [(seed_to_team[SEED_ORDER[i]], seed_to_team[SEED_ORDER[i + 1]])
            for i in range(0, 16, 2)
            if SEED_ORDER[i] in seed_to_team and SEED_ORDER[i + 1] in seed_to_team]


def sim_game(a, b, elos, deterministic=False):
    ea, eb = elos.get(a, STARTING_ELO), elos.get(b, STARTING_ELO)
    p = expected_score(ea, eb)
    return a if (deterministic and p >= 0.5) or (not deterministic and random.random() < p) else b


def sim_region(region_teams, elos, deterministic=False):
    pairs   = first_round_pairs(region_teams)
    winners = [sim_game(a, b, elos, deterministic) for a, b in pairs]
    rounds  = [winners[:]]
    while len(winners) > 1:
        nxt = [sim_game(winners[i], winners[i + 1], elos, deterministic)
               for i in range(0, len(winners) - 1, 2)]
        winners = nxt
        rounds.append(winners[:])
    return winners[0], rounds


def sim_bracket(bracket, elos, deterministic=False):
    region_champs, all_results = {}, {}
    for region, teams in bracket.items():
        champ, rounds = sim_region(teams, elos, deterministic)
        region_champs[region] = champ
        all_results[region]   = rounds

    regions = sorted(region_champs)
    if len(regions) >= 4:
        sf1   = sim_game(region_champs[regions[0]], region_champs[regions[1]], elos, deterministic)
        sf2   = sim_game(region_champs[regions[2]], region_champs[regions[3]], elos, deterministic)
        champ = sim_game(sf1, sf2, elos, deterministic)
        all_results["FinalFour"] = [[sf1, sf2], [champ]]
    else:
        champ = list(region_champs.values())[0]

    return champ, all_results, region_champs


def run_monte_carlo(bracket, elos, n=N_SIMULATIONS):
    champ_counts = defaultdict(int)
    ff_counts    = defaultdict(int)
    for _ in range(n):
        champ, _, rc = sim_bracket(bracket, elos, deterministic=False)
        champ_counts[champ] += 1
        for rc_team in rc.values():
            ff_counts[rc_team] += 1
    return {"champion_counts": dict(champ_counts), "final_four_counts": dict(ff_counts)}


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6: Printable Bracket Visualization
# ══════════════════════════════════════════════════════════════════════════════

def draw_bracket(bracket, elos, results, id_to_name, model_name, color, filename):
    fig, axes = plt.subplots(1, len(bracket), figsize=(5.5 * len(bracket), 16))
    if len(bracket) == 1:
        axes = [axes]

    slot_y  = np.linspace(15.5, 0.5, 16)
    round_x = [0.3, 3.0, 5.2, 7.0, 8.5]

    def draw_box(ax, x, y, team_id, seed, is_winner):
        name = id_to_name.get(team_id, f"T{team_id}")[:14]
        elo  = elos.get(team_id, STARTING_ELO)
        bg   = color if is_winner else "#f5f5f5"
        tc   = "white" if is_winner else "#1a1a1a"
        ec   = color if is_winner else "#cccccc"
        lw   = 2.0 if is_winner else 0.7
        box  = FancyBboxPatch((x, y - 0.3), 2.3, 0.58,
                              boxstyle="round,pad=0.06",
                              facecolor=bg, edgecolor=ec, linewidth=lw, zorder=2)
        ax.add_patch(box)
        ax.text(x + 0.12, y, f"{seed}", fontsize=6, va="center", ha="left",
                color=tc, fontweight="bold", zorder=3)
        ax.text(x + 0.40, y, name, fontsize=6.5, va="center", ha="left", color=tc, zorder=3)
        ax.text(x + 2.25, y, f"{elo:.0f}", fontsize=5.5, va="center", ha="right",
                color=tc, alpha=0.65, zorder=3)

    for ax, (region, region_teams) in zip(axes, sorted(bracket.items())):
        ax.set_xlim(0, 11); ax.set_ylim(-0.5, 17); ax.axis("off")
        ax.set_title(REGION_FULL.get(region, region), fontsize=12,
                     fontweight="bold", color=color, pad=12)

        rnd_labels = ["Round 1", "Round 32", "Sweet 16", "Elite 8"]
        for rx, rl in zip(round_x[:4], rnd_labels):
            ax.text(rx + 1.15, 16.6, rl, ha="center", va="center",
                    fontsize=7, color="#999", style="italic")

        region_results = results.get(region, [])
        seed_to_team   = {s: t for s, t in region_teams}
        pairs = [(SEED_ORDER[i], SEED_ORDER[i + 1]) for i in range(0, 16, 2)]
        r1_winners = region_results[0] if region_results else []

        for i, (s1, s2) in enumerate(pairs):
            t1, t2 = seed_to_team.get(s1), seed_to_team.get(s2)
            y1, y2 = slot_y[i * 2], slot_y[i * 2 + 1]
            w = r1_winners[i] if i < len(r1_winners) else None
            if t1:
                draw_box(ax, round_x[0], y1, t1, s1, t1 == w)
            if t2:
                draw_box(ax, round_x[0], y2, t2, s2, t2 == w)
            mid = (y1 + y2) / 2
            ax.plot([round_x[0] + 2.3, round_x[0] + 2.3], [y1, y2], color="#ddd", lw=0.7)
            ax.plot([round_x[0] + 2.3, round_x[1]], [mid, mid], color="#ddd", lw=0.7)

        for rnd_idx in range(1, min(4, len(region_results))):
            this_round = region_results[rnd_idx]
            prev_round = region_results[rnd_idx - 1]
            n_prev = len(prev_round)
            chunk  = 16 // (2 ** rnd_idx)

            for g in range(0, n_prev, 2):
                t1  = prev_round[g] if g < n_prev else None
                t2  = prev_round[g + 1] if g + 1 < n_prev else None
                i1  = (g // 2) * chunk * 2
                i2  = i1 + chunk * 2 - 1
                y1  = slot_y[min(i1, 15)]
                y2  = slot_y[min(i2, 15)]
                mid = (y1 + y2) / 2
                w   = this_round[g // 2] if g // 2 < len(this_round) else None
                st1 = next((s for s, t in seed_to_team.items() if t == t1), "?")
                st2 = next((s for s, t in seed_to_team.items() if t == t2), "?")
                if t1:
                    draw_box(ax, round_x[rnd_idx], y1, t1, st1, t1 == w)
                if t2:
                    draw_box(ax, round_x[rnd_idx], y2, t2, st2, t2 == w)
                ax.plot([round_x[rnd_idx] + 2.3, round_x[rnd_idx] + 2.3],
                        [y1, y2], color="#ddd", lw=0.7)
                if rnd_idx + 1 < len(round_x):
                    ax.plot([round_x[rnd_idx] + 2.3, round_x[rnd_idx + 1]],
                            [mid, mid], color="#ddd", lw=0.7)

    champ_id = None
    if "FinalFour" in results and results["FinalFour"] and results["FinalFour"][-1]:
        champ_id = results["FinalFour"][-1][0]
    champ_name = id_to_name.get(champ_id, "???") if champ_id else "???"

    fig.suptitle(
        f"🏀  2025 March Madness Bracket  —  {model_name}\n"
        f"🏆  Champion:  {champ_name}  |  Ratings: 2025 season only",
        fontsize=14, fontweight="bold", color=color, y=1.01,
    )
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {filename}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7: Comparison Charts
# ══════════════════════════════════════════════════════════════════════════════

def plot_championship_probs(mc_results, id_to_name, models_config, n=N_SIMULATIONS):
    all_teams = set()
    for name, _, _ in models_config:
        top = sorted(mc_results[name]["champion_counts"].items(), key=lambda x: -x[1])[:15]
        all_teams.update(t for t, _ in top)

    teams_sorted = sorted(
        all_teams,
        key=lambda t: -mc_results[models_config[2][0]]["champion_counts"].get(t, 0),
    )
    labels = [id_to_name.get(t, str(t)) for t in teams_sorted]
    x = np.arange(len(teams_sorted))
    w = 0.27

    fig, ax = plt.subplots(figsize=(16, 6))
    for i, (name, _, color) in enumerate(models_config):
        probs = [mc_results[name]["champion_counts"].get(t, 0) / n * 100 for t in teams_sorted]
        ax.bar(x + i * w - w, probs, w, label=name, color=color, alpha=0.85,
               edgecolor="white", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("Championship Probability (%)")
    ax.set_title(
        f"2025 March Madness — Championship Probabilities\n"
        f"({n:,} Monte Carlo simulations × 3 models  |  Ratings based on 2025 season only)",
        fontsize=13, fontweight="bold",
    )
    ax.legend(fontsize=10)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.15)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig("championship_probs_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: championship_probs_comparison.png")


def plot_elo_distributions(tourney_teams, elo_list, models_config):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    for ax, (name, _, color), elos in zip(axes, models_config, elo_list):
        vals = [elos.get(t, STARTING_ELO) for t in tourney_teams]
        ax.hist(vals, bins=20, color=color, alpha=0.8, edgecolor="white")
        ax.axvline(np.mean(vals), color="red", linestyle="--", linewidth=1.5,
                   label=f"Mean: {np.mean(vals):.0f}")
        ax.set_title(name, fontweight="bold")
        ax.set_xlabel("ELO Rating"); ax.set_ylabel("# Teams")
        ax.legend(fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.suptitle("ELO Rating Distributions — 2025 Tournament Teams (2025 season only)",
                 fontweight="bold")
    plt.tight_layout()
    plt.savefig("elo_distributions.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: elo_distributions.png")


def plot_model_correlation(tourney_teams, elos_v1, elos_v2, elos_v3):
    common = [t for t in tourney_teams if t in elos_v1 and t in elos_v2 and t in elos_v3]
    v1 = [elos_v1[t] for t in common]
    v2 = [elos_v2[t] for t in common]
    v3 = [elos_v3[t] for t in common]

    pairs = [
        (v1, v2, "V1 Basic", "V2 Enhanced"),
        (v1, v3, "V1 Basic", "V3 Kitchen Sink"),
        (v2, v3, "V2 Enhanced", "V3 Kitchen Sink"),
    ]
    colors = ["#3b82f6", "#22c55e", "#d97706"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (xa, xb, la, lb), color in zip(axes, pairs, colors):
        corr = np.corrcoef(xa, xb)[0, 1]
        ax.scatter(xa, xb, c=color, alpha=0.7, edgecolors="white", s=55, linewidth=0.4)
        m, b = np.polyfit(xa, xb, 1)
        xs = np.linspace(min(xa), max(xa), 100)
        ax.plot(xs, m * xs + b, "k--", linewidth=1, alpha=0.4)
        ax.set_xlabel(la, fontsize=9); ax.set_ylabel(lb, fontsize=9)
        ax.set_title(f"r = {corr:.3f}", fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.suptitle("Model Agreement — ELO Rating Correlations (2025 season)", fontweight="bold")
    plt.tight_layout()
    plt.savefig("model_correlation.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: model_correlation.png")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8: Results Summary Table
# ══════════════════════════════════════════════════════════════════════════════

def print_summary_table(mc_results, id_to_name, models_config, n=N_SIMULATIONS):
    names = [name for name, _, _ in models_config]
    w = 70

    for label, count_key in [("FINAL FOUR PROBABILITY", "final_four_counts"),
                              ("CHAMPIONSHIP PROBABILITY", "champion_counts")]:
        print("=" * w)
        print(f" {label}  (2025 season ratings)")
        print("=" * w)
        print(f"{'Team':<24}" + "".join(f"{n:>15}" for n in names))
        print("-" * w)

        all_teams = set()
        for name, _, _ in models_config:
            top = sorted(mc_results[name][count_key].items(), key=lambda x: -x[1])[:15]
            all_teams.update(t for t, _ in top)

        sorted_teams = sorted(
            all_teams,
            key=lambda t: -mc_results[names[-1]][count_key].get(t, 0),
        )
        for tid in sorted_teams[:16]:
            team_name = id_to_name.get(tid, str(tid))[:23]
            row = f"{team_name:<24}"
            for name, _, _ in models_config:
                pct = mc_results[name][count_key].get(tid, 0) / n * 100
                row += f"{pct:>14.1f}%"
            print(row)
        print()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 9: Main Execution
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  🏀 March Madness ELO Bracket Predictor")
    print(f"  Calibration: {CALIBRATION_START}-{CALIBRATION_END} | Final ratings: {CURRENT_SEASON} only")
    print("=" * 60)

    teams_df, reg_df, tourney_df, seeds_df, conf_df = load_data()
    id_to_name = dict(zip(teams_df["TeamID"], teams_df["TeamName"]))

    # ── Train all 3 models (two-phase) ────────────────────────────────────────
    print("Training models (Phase 1: calibration | Phase 2: 2025 season)...")
    print("  [1/3] Basic ELO...")
    elos_v1 = train_basic_elo(reg_df, tourney_df)

    print("  [2/3] Enhanced ELO...")
    elos_v2 = train_enhanced_elo(reg_df, tourney_df)

    print("  [3/3] Kitchen Sink ELO...")
    elos_v3_raw = train_kitchen_sink_elo(reg_df, tourney_df, conf_df)
    elos_v3     = blend_seed_prior(elos_v3_raw, seeds_df, PREDICT_YEAR)
    print()

    models_config = [
        ("V1 Basic",        elos_v1, "#3b82f6"),
        ("V2 Enhanced",     elos_v2, "#22c55e"),
        ("V3 Kitchen Sink", elos_v3, "#d97706"),
    ]

    # ── Sanity check: top 10 per model ────────────────────────────────────────
    print("Top 10 ELO ratings by model (2025 season only):")
    for name, elos, _ in models_config:
        top10 = sorted(elos.items(), key=lambda x: -x[1])[:10]
        print(f"\n  {name}:")
        for i, (tid, elo) in enumerate(top10, 1):
            print(f"    {i:2d}. {id_to_name.get(tid, tid):<25} {elo:.1f}")
    print()

    # ── Parse bracket ─────────────────────────────────────────────────────────
    bracket_2025  = parse_bracket(seeds_df, PREDICT_YEAR)
    tourney_teams = seeds_df[seeds_df["Season"] == PREDICT_YEAR]["TeamID"].astype(int).tolist()
    print(f"Bracket parsed: {len(tourney_teams)} teams across {len(bracket_2025)} regions\n")

    # ── Monte Carlo ───────────────────────────────────────────────────────────
    mc_results = {}
    print(f"Running Monte Carlo simulations ({N_SIMULATIONS:,} × 3 models)...")
    for name, elos, color in models_config:
        champ_det, _, _ = sim_bracket(bracket_2025, elos, deterministic=True)
        mc = run_monte_carlo(bracket_2025, elos, N_SIMULATIONS)
        mc_results[name] = mc

        top5 = sorted(mc["champion_counts"].items(), key=lambda x: -x[1])[:5]
        print(f"\n  {name} — Deterministic champion: {id_to_name.get(champ_det, champ_det)}")
        print(f"  Monte Carlo top 5:")
        for tid, cnt in top5:
            pct = cnt / N_SIMULATIONS * 100
            print(f"    {id_to_name.get(tid, tid):<25} {pct:5.1f}%")
    print()

    # ── Draw brackets ─────────────────────────────────────────────────────────
    print("Drawing brackets...")
    for name, elos, color in models_config:
        _, results, _ = sim_bracket(bracket_2025, elos, deterministic=True)
        safe_name = name.lower().replace(" ", "_")
        draw_bracket(bracket_2025, elos, results, id_to_name, name, color,
                     f"bracket_{safe_name}.png")

    # ── Comparison charts ─────────────────────────────────────────────────────
    print("Generating comparison charts...")
    plot_championship_probs(mc_results, id_to_name, models_config)
    plot_elo_distributions(tourney_teams, [e for _, e, _ in models_config], models_config)
    plot_model_correlation(tourney_teams, elos_v1, elos_v2, elos_v3)

    # ── Summary tables ────────────────────────────────────────────────────────
    print()
    print_summary_table(mc_results, id_to_name, models_config)

    print("=" * 60)
    print("  ✅ Done! Files saved:")
    print("     bracket_v1_basic.png")
    print("     bracket_v2_enhanced.png")
    print("     bracket_v3_kitchen_sink.png")
    print("     championship_probs_comparison.png")
    print("     elo_distributions.png")
    print("     model_correlation.png")
    print("=" * 60)


if __name__ == "__main__":
    main()
