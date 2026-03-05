"""
╔══════════════════════════════════════════════════════════════════════════════╗
║          🏀 MARCH MADNESS ELO BACKTEST — 2022, 2023, 2024                  ║
║                                                                              ║
║  For each backtest year:                                                     ║
║    - Calibrate on all seasons BEFORE that year                              ║
║    - Hard reset, run that year's regular season only                        ║
║    - Apply SOS post-processing                                               ║
║    - Predict each tournament game and compare to actual result              ║
║                                                                              ║
║  Metrics:                                                                    ║
║    1. Game-by-game accuracy vs always-pick-higher-seed baseline             ║
║    2. Calibration curve — does predicted 70% actually win 70%?              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import sys, os, argparse
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import defaultdict
import warnings
import random
warnings.filterwarnings("ignore")

# ── Import shared logic from main predictor ───────────────────────────────────
from march_madness_elo import (
    # Config
    STARTING_ELO, K_FACTOR, K_FACTOR_ENHANCED, HOME_ADVANTAGE,
    REVERSION_FACTOR, RECENCY_HALF_LIFE, SEED_PRIOR_WEIGHT,
    CONF_STRENGTH_W, SOS_SHRINKAGE, SOS_IMPACT,
    SOS_MIN_MULTIPLIER, SOS_MAX_MULTIPLIER, SOS_SCALE,
    SEED_ELO_MAP, CONF_STRENGTH,
    # Utilities
    expected_score, mov_multiplier, recency_weight,
    revert_to_mean, games_for_seasons, games_for_season_reg_only,
    # Training
    _run_elo_pass, _hard_reset, _basic_update, _enhanced_update,
    _make_kitchen_sink_update, build_conf_adjustments,
    blend_seed_prior, apply_sos_adjustment,
    # SOS
    RollingSOS,
)

# ── Backtest Config ───────────────────────────────────────────────────────────
BACKTEST_YEARS    = [2022, 2023, 2024]
CALIBRATION_START = 2010
DATA_DIR          = os.path.join(os.path.dirname(__file__), "data")

# Calibration bins for the calibration curve
N_BINS = 10

# ── Parameter sweep config ────────────────────────────────────────────────────
SWEEP_PARAMS = {
    "seed_prior_weight": [0.0, 0.05, 0.10, 0.15, 0.20],
    "sos_scale":         [20, 30, 40, 50, 60],
    "sos_shrinkage":     [5, 10, 15, 20],
}


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1: Per-Year Model Training
# ══════════════════════════════════════════════════════════════════════════════

def train_for_year(year, reg_df, tourney_df, conf_df, seeds_df):
    """
    Train all 3 models as if it's the eve of {year}'s tournament:
      - Calibration: CALIBRATION_START to year-1
      - Current season: {year} regular season only
      - SOS post-processing normalized to {year} tournament field
    Returns:
      { "V1 Basic": elos, "V2 Enhanced": elos, "V3 Kitchen Sink": elos }
      and sos_scores dict for V2/V3
    """
    cal_seasons  = list(range(CALIBRATION_START, year))   # e.g. 2010-2023 for 2024
    tourney_ids  = seeds_df[seeds_df["Season"] == year]["TeamID"].astype(int).tolist()

    # ── V1: Basic ─────────────────────────────────────────────────────────────
    elos_v1 = defaultdict(lambda: STARTING_ELO)
    cal_games = games_for_seasons(reg_df, tourney_df, cal_seasons)
    elos_v1 = _run_elo_pass(cal_games, elos_v1, _basic_update)
    elos_v1 = _hard_reset(elos_v1)
    for _, g in games_for_season_reg_only(reg_df, year).iterrows():
        wd, ld = _basic_update(g, elos_v1)
        elos_v1[int(g["WTeamID"])] += wd
        elos_v1[int(g["LTeamID"])] += ld
    elos_v1 = dict(elos_v1)

    # ── V2: Enhanced + SOS ────────────────────────────────────────────────────
    elos_v2 = defaultdict(lambda: STARTING_ELO)
    elos_v2 = _run_elo_pass(cal_games, elos_v2, _enhanced_update)
    elos_v2 = _hard_reset(elos_v2)
    sos_v2  = RollingSOS()
    for _, g in games_for_season_reg_only(reg_df, year).iterrows():
        wd, ld = _enhanced_update(g, elos_v2, sos=sos_v2)
        elos_v2[int(g["WTeamID"])] += wd
        elos_v2[int(g["LTeamID"])] += ld
        sos_v2.record_result(int(g["WTeamID"]), int(g["LTeamID"]))
    sos_scores_v2 = sos_v2.final_sos_scores()
    elos_v2 = apply_sos_adjustment(dict(elos_v2), sos_scores_v2, tourney_ids)

    # ── V3: Kitchen Sink + SOS ────────────────────────────────────────────────
    elos_v3 = defaultdict(lambda: STARTING_ELO)
    cur_season_cal, conf_adj = None, {}
    for _, g in cal_games.iterrows():
        if g["Season"] != cur_season_cal:
            if cur_season_cal is not None:
                rv = revert_to_mean(dict(elos_v3))
                elos_v3.clear(); elos_v3.update(rv)
            cur_season_cal = g["Season"]
            conf_adj = build_conf_adjustments(conf_df, int(g["Season"]))
        wd, ld = _make_kitchen_sink_update(conf_adj)(g, elos_v3)
        elos_v3[int(g["WTeamID"])] += wd
        elos_v3[int(g["LTeamID"])] += ld

    elos_v3   = _hard_reset(elos_v3)
    sos_v3    = RollingSOS()
    conf_2025 = build_conf_adjustments(conf_df, year)
    upd_fn    = _make_kitchen_sink_update(conf_2025)
    for _, g in games_for_season_reg_only(reg_df, year).iterrows():
        wd, ld = upd_fn(g, elos_v3, sos=sos_v3)
        elos_v3[int(g["WTeamID"])] += wd
        elos_v3[int(g["LTeamID"])] += ld
        sos_v3.record_result(int(g["WTeamID"]), int(g["LTeamID"]))
    sos_scores_v3 = sos_v3.final_sos_scores()
    elos_v3_sos   = apply_sos_adjustment(dict(elos_v3), sos_scores_v3, tourney_ids)
    elos_v3_final = blend_seed_prior(elos_v3_sos, seeds_df, year)

    return {
        "V1 Basic":        elos_v1,
        "V2 Enhanced":     elos_v2,
        "V3 Kitchen Sink": elos_v3_final,
    }


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2: Prediction & Accuracy
# ══════════════════════════════════════════════════════════════════════════════

ROUND_MAP = {
    # DayNum ranges for NCAA tournament rounds (approximate, varies slightly by year)
    # Round of 64 / First Four: 134-136
    # Round of 32: 137-138
    # Sweet 16: 143-144
    # Elite 8: 145-146
    # Final Four: 152
    # Championship: 154
}

def day_to_round(day):
    if day <= 136:  return (1, "First Round")
    if day <= 138:  return (2, "Second Round")
    if day <= 144:  return (3, "Sweet 16")
    if day <= 146:  return (4, "Elite 8")
    if day <= 152:  return (5, "Final Four")
    return              (6, "Championship")


def predict_games(year, elos_dict, tourney_df, seeds_df):
    """
    For each actual tournament game in {year}, compute:
      - predicted winner (higher ELO)
      - predicted win probability for the actual winner
      - whether the prediction was correct
      - seed matchup
    Returns a list of result dicts, one per model per game.
    """
    games = tourney_df[tourney_df["Season"] == year].copy()

    # Build seed lookup
    s = seeds_df[seeds_df["Season"] == year].copy()
    s["seed_num"] = s["Seed"].str[1:3].str.replace("a","").str.replace("b","").astype(int)
    seed_map = dict(zip(s["TeamID"].astype(int), s["seed_num"]))

    records = []
    for _, g in games.iterrows():
        winner = int(g["WTeamID"])
        loser  = int(g["LTeamID"])
        round_num, round_name = day_to_round(g["DayNum"])
        seed_w = seed_map.get(winner, 8)
        seed_l = seed_map.get(loser,  8)

        # Seed baseline: lower seed number = higher seed = pick them
        seed_pick_correct = int(seed_w < seed_l)   # 1 if higher seed won

        for model_name, elos in elos_dict.items():
            ew = elos.get(winner, STARTING_ELO)
            el = elos.get(loser,  STARTING_ELO)
            # Win probability for the actual winner
            p_winner = expected_score(ew, el)
            predicted_correct = int(p_winner >= 0.5)

            records.append({
                "year":              year,
                "round":             round_num,
                "round_name":        round_name,
                "model":             model_name,
                "winner":            winner,
                "loser":             loser,
                "seed_winner":       seed_w,
                "seed_loser":        seed_l,
                "elo_winner":        round(ew, 1),
                "elo_loser":         round(el, 1),
                "p_winner":          round(p_winner, 4),
                "predicted_correct": predicted_correct,
                "seed_pick_correct": seed_pick_correct,
            })

    return records


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3: Calibration
# ══════════════════════════════════════════════════════════════════════════════

def compute_calibration(records_df, model_name, n_bins=N_BINS):
    """
    Bin predictions by predicted win probability and compute actual win rate
    per bin. Returns (bin_midpoints, actual_win_rates, bin_counts).

    Note: we symmetrize — for each game we add both p_winner and (1-p_winner)
    so the calibration covers the full [0,1] range, not just >0.5.
    """
    df = records_df[records_df["model"] == model_name].copy()

    probs, actuals = [], []
    for _, row in df.iterrows():
        probs.append(row["p_winner"])
        actuals.append(1)                    # actual winner won
        probs.append(1 - row["p_winner"])
        actuals.append(0)                    # actual loser lost

    probs   = np.array(probs)
    actuals = np.array(actuals)
    bins    = np.linspace(0, 1, n_bins + 1)
    mids, win_rates, counts = [], [], []

    for i in range(n_bins):
        mask = (probs >= bins[i]) & (probs < bins[i + 1])
        if mask.sum() > 0:
            mids.append((bins[i] + bins[i + 1]) / 2)
            win_rates.append(actuals[mask].mean())
            counts.append(mask.sum())

    return np.array(mids), np.array(win_rates), np.array(counts)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4: Plotting
# ══════════════════════════════════════════════════════════════════════════════

MODEL_COLORS = {
    "V1 Basic":        "#3b82f6",
    "V2 Enhanced":     "#22c55e",
    "V3 Kitchen Sink": "#d97706",
    "Seed Baseline":   "#9ca3af",
}


def plot_accuracy_by_round(records_df, output_path="backtest_accuracy_by_round.png"):
    """
    Grouped bar chart: accuracy by round for each model + seed baseline.
    """
    models   = ["V1 Basic", "V2 Enhanced", "V3 Kitchen Sink"]
    rounds   = sorted(records_df["round"].unique())
    round_labels = [records_df[records_df["round"] == r]["round_name"].iloc[0] for r in rounds]

    fig, axes = plt.subplots(1, len(BACKTEST_YEARS) + 1, figsize=(18, 5),
                             gridspec_kw={"width_ratios": [1] * len(BACKTEST_YEARS) + [1.2]})

    x = np.arange(len(rounds))
    w = 0.18

    def _acc(df, model, round_num):
        sub = df[(df["model"] == model) & (df["round"] == round_num)]
        if len(sub) == 0: return 0
        return sub["predicted_correct"].mean() * 100

    def _seed_acc(df, round_num):
        sub = df[(df["model"] == "V1 Basic") & (df["round"] == round_num)]
        if len(sub) == 0: return 0
        return sub["seed_pick_correct"].mean() * 100

    for ax_idx, (ax, year) in enumerate(zip(axes[:-1], BACKTEST_YEARS)):
        df_y = records_df[records_df["year"] == year]
        for i, model in enumerate(models):
            vals = [_acc(df_y, model, r) for r in rounds]
            ax.bar(x + (i - 1.5) * w, vals, w, label=model,
                   color=MODEL_COLORS[model], alpha=0.85, edgecolor="white")
        # Seed baseline
        seed_vals = [_seed_acc(df_y, r) for r in rounds]
        ax.bar(x + 1.5 * w, seed_vals, w, label="Seed Baseline",
               color=MODEL_COLORS["Seed Baseline"], alpha=0.85, edgecolor="white")

        ax.set_title(f"{year} Tournament", fontweight="bold")
        ax.set_xticks(x); ax.set_xticklabels(round_labels, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("% Games Correct"); ax.set_ylim(0, 100)
        ax.axhline(50, color="red", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.grid(axis="y", alpha=0.3)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    # Combined accuracy across all years
    ax = axes[-1]
    for i, model in enumerate(models):
        vals = [_acc(records_df, model, r) for r in rounds]
        ax.bar(x + (i - 1.5) * w, vals, w, label=model,
               color=MODEL_COLORS[model], alpha=0.85, edgecolor="white")
    seed_vals = [_seed_acc(records_df, r) for r in rounds]
    ax.bar(x + 1.5 * w, seed_vals, w, label="Seed Baseline",
           color=MODEL_COLORS["Seed Baseline"], alpha=0.85, edgecolor="white")

    ax.set_title("2022–2024 Combined", fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(round_labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylim(0, 100)
    ax.axhline(50, color="red", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    fig.suptitle("Backtest: % Tournament Games Predicted Correctly by Round\n"
                 "vs Seed Baseline (always pick higher seed)",
                 fontweight="bold", fontsize=13)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {output_path}")


def plot_calibration_curves(records_df, output_path="backtest_calibration.png"):
    """
    Calibration curves: predicted win probability vs actual win rate.
    Perfect calibration = diagonal line.
    One subplot per model, all years combined.
    """
    models = ["V1 Basic", "V2 Enhanced", "V3 Kitchen Sink"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, model in zip(axes, models):
        mids, win_rates, counts = compute_calibration(records_df, model)
        color = MODEL_COLORS[model]

        # Perfect calibration line
        ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.4, label="Perfect calibration")

        # Calibration curve — size dots by sample count
        sizes = np.clip(counts * 3, 30, 300)
        ax.scatter(mids, win_rates, s=sizes, color=color, alpha=0.85,
                   edgecolors="white", linewidth=0.5, zorder=3)
        ax.plot(mids, win_rates, color=color, linewidth=2, alpha=0.7)

        # Shade the error region
        ax.fill_between(mids, win_rates, mids,
                        alpha=0.08, color=color)

        # Compute Brier score (lower = better)
        df_m = records_df[records_df["model"] == model]
        brier = np.mean((df_m["p_winner"] - 1) ** 2)   # actual=1 always (we stored p_winner)
        # Symmetrized Brier
        brier_sym = np.mean(
            [(p - 1)**2 for p in df_m["p_winner"]] +
            [(1 - p - 0)**2 for p in df_m["p_winner"]]
        ) / 2

        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel("Predicted Win Probability", fontsize=10)
        ax.set_ylabel("Actual Win Rate", fontsize=10)
        ax.set_title(f"{model}\nBrier score: {brier_sym:.4f}", fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    fig.suptitle("Calibration Curves — 2022–2024 Tournament Games Combined\n"
                 "(dot size = number of predictions in bin, perfect model follows diagonal)",
                 fontweight="bold", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {output_path}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5: Summary Table
# ══════════════════════════════════════════════════════════════════════════════

def print_summary(records_df):
    models = ["V1 Basic", "V2 Enhanced", "V3 Kitchen Sink"]
    w = 72

    print()
    print("=" * w)
    print("  BACKTEST SUMMARY — 2022, 2023, 2024 NCAA Tournament")
    print("=" * w)

    # ── Overall accuracy ──────────────────────────────────────────────────────
    print(f"\n  {'Model':<22} {'Overall':>9}  {'R1':>6}  {'R2':>6}  "
          f"{'S16':>6}  {'E8':>6}  {'FF':>6}  {'Champ':>6}")
    print(f"  {'-' * 68}")

    rounds = sorted(records_df["round"].unique())

    def acc(df, model, rnd=None):
        sub = df[df["model"] == model]
        if rnd: sub = sub[sub["round"] == rnd]
        if len(sub) == 0: return float("nan")
        return sub["predicted_correct"].mean() * 100

    def seed_acc(df, rnd=None):
        sub = df[df["model"] == "V1 Basic"]
        if rnd: sub = sub[sub["round"] == rnd]
        if len(sub) == 0: return float("nan")
        return sub["seed_pick_correct"].mean() * 100

    for model in models:
        row = f"  {model:<22} {acc(records_df, model):>8.1f}%"
        for r in rounds:
            v = acc(records_df, model, r)
            row += f"  {v:>5.1f}%"
        print(row)

    # Seed baseline
    row = f"  {'Seed Baseline':<22} {seed_acc(records_df):>8.1f}%"
    for r in rounds:
        row += f"  {seed_acc(records_df, r):>5.1f}%"
    print(row)

    # ── Per-year breakdown ────────────────────────────────────────────────────
    print(f"\n  {'Model':<22} {'2022':>8}  {'2023':>8}  {'2024':>8}  {'Combined':>10}")
    print(f"  {'-' * 62}")
    for model in models:
        row = f"  {model:<22}"
        for year in BACKTEST_YEARS:
            df_y = records_df[records_df["year"] == year]
            row += f"  {acc(df_y, model):>7.1f}%"
        row += f"  {acc(records_df, model):>9.1f}%"
        print(row)
    row = f"  {'Seed Baseline':<22}"
    for year in BACKTEST_YEARS:
        df_y = records_df[records_df["year"] == year]
        row += f"  {seed_acc(df_y):>7.1f}%"
    row += f"  {seed_acc(records_df):>9.1f}%"
    print(row)

    # ── Upsets: how often did models correctly predict upsets ─────────────────
    print(f"\n  UPSET PREDICTION (games where lower seed won)")
    print(f"  {'-' * 62}")
    upsets = records_df[(records_df["model"] == "V1 Basic") &
                        (records_df["seed_winner"] > records_df["seed_loser"])]
    n_upsets = len(upsets)
    print(f"  Total upsets in 2022-2024: {n_upsets} of "
          f"{len(records_df[records_df['model']=='V1 Basic'])} games "
          f"({n_upsets/len(records_df[records_df['model']=='V1 Basic'])*100:.1f}%)")

    for model in models:
        df_m   = records_df[records_df["model"] == model]
        ups    = df_m[df_m["seed_winner"] > df_m["seed_loser"]]
        called = ups["predicted_correct"].sum()
        print(f"  {model:<22}  correctly predicted {called}/{n_upsets} upsets "
              f"({called/n_upsets*100:.1f}%)")

    print()
    print("  Note: 'correctly predicted upset' = model picked the lower-seeded")
    print("  team to win based on ELO differential before the game.")
    print()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6: Parameter Sweep
# ══════════════════════════════════════════════════════════════════════════════

def train_for_year_with_params(year, reg_df, tourney_df, conf_df, seeds_df,
                                seed_prior_weight, sos_scale, sos_shrinkage):
    """
    Variant of train_for_year that accepts overrideable hyperparameters.
    Only trains V3 Kitchen Sink since that's what we're sweeping.
    Returns (elos_v3, sos_scores_v3)
    """
    import march_madness_elo as mme

    cal_seasons = list(range(CALIBRATION_START, year))
    tourney_ids = seeds_df[seeds_df["Season"] == year]["TeamID"].astype(int).tolist()
    cal_games   = games_for_seasons(reg_df, tourney_df, cal_seasons)

    # V3 calibration pass
    elos = defaultdict(lambda: STARTING_ELO)
    cur_season_cal, conf_adj = None, {}
    for _, g in cal_games.iterrows():
        if g["Season"] != cur_season_cal:
            if cur_season_cal is not None:
                rv = revert_to_mean(dict(elos))
                elos.clear(); elos.update(rv)
            cur_season_cal = g["Season"]
            conf_adj = build_conf_adjustments(conf_df, int(g["Season"]))
        wd, ld = _make_kitchen_sink_update(conf_adj)(g, elos)
        elos[int(g["WTeamID"])] += wd
        elos[int(g["LTeamID"])] += ld

    # Phase 2: current season with custom shrinkage
    elos     = _hard_reset(elos)
    sos      = RollingSOS()
    sos._shrinkage_override = sos_shrinkage   # monkey-patch for sweep
    conf_cur = build_conf_adjustments(conf_df, year)
    upd_fn   = _make_kitchen_sink_update(conf_cur)

    for _, g in games_for_season_reg_only(reg_df, year).iterrows():
        wd, ld = upd_fn(g, elos, sos=sos)
        elos[int(g["WTeamID"])] += wd
        elos[int(g["LTeamID"])] += ld
        sos.record_result(int(g["WTeamID"]), int(g["LTeamID"]))

    sos_scores = sos.final_sos_scores()

    # Post-processing with custom scale
    sos_vals  = np.array([sos_scores[t] for t in tourney_ids if t in sos_scores])
    sos_mean  = sos_vals.mean()
    sos_std   = sos_vals.std() if sos_vals.std() > 0 else 1.0
    adjusted  = dict(elos)
    for tid in [t for t in elos if t in sos_scores]:
        sos_z = (sos_scores[tid] - sos_mean) / sos_std
        adjusted[tid] = elos[tid] + sos_z * sos_scale

    # Seed prior with custom weight
    s = seeds_df[seeds_df["Season"] == year]
    for _, row in s.iterrows():
        tid = int(row["TeamID"])
        try:
            seed_num = int("".join(filter(str.isdigit, str(row["Seed"])[:3])))
        except Exception:
            seed_num = 8
        seed_elo = SEED_ELO_MAP.get(seed_num, STARTING_ELO)
        current  = adjusted.get(tid, STARTING_ELO)
        adjusted[tid] = (1 - seed_prior_weight) * current + seed_prior_weight * seed_elo

    return adjusted


def run_sweep(reg_df, tourney_df, conf_df, seeds_df, tourney_df_actual):
    """
    Sweep seed_prior_weight, sos_scale, and sos_shrinkage independently,
    holding the other two at their default values. Reports overall accuracy
    and upset detection rate for each value.
    """
    DEFAULTS = {
        "seed_prior_weight": 0.15,
        "sos_scale":         40,
        "sos_shrinkage":     10,
    }

    def score_params(spw, ss, sh):
        records = []
        for year in BACKTEST_YEARS:
            elos = train_for_year_with_params(
                year, reg_df, tourney_df, conf_df, seeds_df,
                seed_prior_weight=spw, sos_scale=ss, sos_shrinkage=sh,
            )
            games = tourney_df_actual[tourney_df_actual["Season"] == year]
            s = seeds_df[seeds_df["Season"] == year].copy()
            s["seed_num"] = s["Seed"].str[1:3].str.replace("a","").str.replace("b","").astype(int)
            seed_map = dict(zip(s["TeamID"].astype(int), s["seed_num"]))

            for _, g in games.iterrows():
                winner = int(g["WTeamID"])
                loser  = int(g["LTeamID"])
                ew = elos.get(winner, STARTING_ELO)
                el = elos.get(loser,  STARTING_ELO)
                p_winner = expected_score(ew, el)
                seed_w = seed_map.get(winner, 8)
                seed_l = seed_map.get(loser,  8)
                records.append({
                    "correct":       int(p_winner >= 0.5),
                    "is_upset":      int(seed_w > seed_l),
                    "upset_correct": int(p_winner >= 0.5 and seed_w > seed_l),
                })

        df = pd.DataFrame(records)
        n_upsets = df["is_upset"].sum()
        return {
            "overall":        df["correct"].mean() * 100,
            "upset_called":   df["upset_correct"].sum() / max(n_upsets, 1) * 100,
        }

    results = []
    for param_name, values in SWEEP_PARAMS.items():
        print(f"\n  Sweeping {param_name}:")
        print(f"  {'Value':>12}  {'Overall Acc':>12}  {'Upset Det':>10}")
        print(f"  {'-'*38}")
        for val in values:
            kwargs = dict(DEFAULTS)
            kwargs[param_name] = val
            scores = score_params(kwargs["seed_prior_weight"],
                                  kwargs["sos_scale"],
                                  kwargs["sos_shrinkage"])
            marker = " ◄ current" if val == DEFAULTS[param_name] else ""
            print(f"  {str(val):>12}  {scores['overall']:>11.1f}%  "
                  f"{scores['upset_called']:>9.1f}%{marker}")
            results.append({
                "param": param_name, "value": val,
                "overall_acc": scores["overall"],
                "upset_det":   scores["upset_called"],
            })

    return pd.DataFrame(results)


def plot_sweep_results(sweep_df, output_path="backtest_sweep.png"):
    """Line plots for each parameter showing accuracy and upset detection vs value."""
    params = sweep_df["param"].unique()
    fig, axes = plt.subplots(2, len(params), figsize=(15, 8))

    DEFAULTS = {"seed_prior_weight": 0.15, "sos_scale": 40, "sos_shrinkage": 10}

    for col, param in enumerate(params):
        df_p  = sweep_df[sweep_df["param"] == param].sort_values("value")
        ax_top = axes[0][col]
        ax_bot = axes[1][col]
        default_val = DEFAULTS.get(param)

        for ax, metric, color, label in [
            (ax_top, "overall_acc",  "#3b82f6", "Overall Accuracy %"),
            (ax_bot, "upset_det",    "#d97706", "Upset Detection %"),
        ]:
            ax.plot(df_p["value"], df_p[metric], "o-", color=color,
                    linewidth=2, markersize=7)
            if default_val is not None:
                ax.axvline(default_val, color="red", linestyle="--",
                           linewidth=1, alpha=0.6, label=f"Current ({default_val})")
            ax.set_xlabel(param.replace("_", " ").title(), fontsize=9)
            ax.set_ylabel(label, fontsize=9)
            ax.set_title(f"{param.replace('_',' ').title()}\nvs {label}",
                         fontsize=9, fontweight="bold")
            ax.grid(alpha=0.3)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.legend(fontsize=8)

    fig.suptitle("Parameter Sweep — V3 Kitchen Sink on 2022–2024 Backtest",
                 fontweight="bold", fontsize=13)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {output_path}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7: Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", action="store_true",
                        help="Run parameter sweep on V3 Kitchen Sink")
    args = parser.parse_args()

    print("=" * 60)
    print("  🏀 March Madness ELO Backtest")
    print(f"  Years: {BACKTEST_YEARS}")
    if args.sweep:
        print("  Mode: PARAMETER SWEEP")
    print("=" * 60)

    # ── Load data ─────────────────────────────────────────────────────────────
    print("Loading data...")
    teams_df   = pd.read_csv(os.path.join(DATA_DIR, "MTeams.csv"))
    reg_df     = pd.read_csv(os.path.join(DATA_DIR, "MRegularSeasonDetailedResults.csv"))
    tourney_df = pd.read_csv(os.path.join(DATA_DIR, "MNCAATourneyDetailedResults.csv"))
    seeds_df   = pd.read_csv(os.path.join(DATA_DIR, "MNCAATourneySeeds.csv"))
    conf_path  = os.path.join(DATA_DIR, "MTeamConferences.csv")
    conf_df    = pd.read_csv(conf_path) if os.path.exists(conf_path) else None
    id_to_name = dict(zip(teams_df["TeamID"], teams_df["TeamName"]))
    print(f"  Loaded: {len(teams_df)} teams, {len(reg_df)} reg games, "
          f"{len(tourney_df)} tourney games\n")

    # ── Train and predict for each year ───────────────────────────────────────
    all_records = []

    for year in BACKTEST_YEARS:
        print(f"  [{year}] Training on {CALIBRATION_START}-{year-1} "
              f"+ {year} regular season...")
        elos_dict = train_for_year(year, reg_df, tourney_df, conf_df, seeds_df)

        n_tourney = len(tourney_df[tourney_df["Season"] == year])
        print(f"  [{year}] Predicting {n_tourney} tournament games...")

        records = predict_games(year, elos_dict, tourney_df, seeds_df)
        all_records.extend(records)

        # Quick per-year accuracy preview
        df_y = pd.DataFrame(records)
        for model in ["V1 Basic", "V2 Enhanced", "V3 Kitchen Sink"]:
            df_m = df_y[df_y["model"] == model]
            acc  = df_m["predicted_correct"].mean() * 100
            print(f"         {model:<22}  {acc:.1f}% correct")
        seed_acc = df_y[df_y["model"] == "V1 Basic"]["seed_pick_correct"].mean() * 100
        print(f"         {'Seed Baseline':<22}  {seed_acc:.1f}% correct")
        print()

    records_df = pd.DataFrame(all_records)

    # ── Print summary table ───────────────────────────────────────────────────
    print_summary(records_df)

    # ── Save records CSV ──────────────────────────────────────────────────────
    records_df.to_csv("backtest_results.csv", index=False)
    print("  Saved: backtest_results.csv")

    # ── Plots ─────────────────────────────────────────────────────────────────
    print("Generating plots...")
    plot_accuracy_by_round(records_df)
    plot_calibration_curves(records_df)

    # ── Parameter sweep (optional) ────────────────────────────────────────────
    if args.sweep:
        print()
        print("=" * 60)
        print("  Running parameter sweep on V3 Kitchen Sink...")
        print("  (sweeping seed_prior_weight, sos_scale, sos_shrinkage)")
        print("=" * 60)
        sweep_df = run_sweep(reg_df, tourney_df, conf_df, seeds_df, tourney_df)
        sweep_df.to_csv("backtest_sweep_results.csv", index=False)
        print("\n  Saved: backtest_sweep_results.csv")
        plot_sweep_results(sweep_df)

    print()
    print("=" * 60)
    print("  ✅ Backtest complete!")
    print("     backtest_results.csv")
    print("     backtest_accuracy_by_round.png")
    print("     backtest_calibration.png")
    if args.sweep:
        print("     backtest_sweep_results.csv")
        print("     backtest_sweep.png")
    print("=" * 60)


if __name__ == "__main__":
    main()
