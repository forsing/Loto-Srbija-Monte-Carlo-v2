"""
MonteCarlo_1_v2 — jaca MC varijanta za Loto 7/39.

Razlika u odnosu na MonteCarlo_v1:
  - seed = 39
  - bez proste frekvencije pojedinacnih brojeva u skoru
  - skor koristi oblik kombinacije, gap distribuciju, parove brojeva i novelty
  - tezine skora bira mini-backtest: validaciona izvlacenja protiv uniformne MC reference
  - MC optimizacija radi u batch-evima, pa moze 1_000_000 kandidata
  - snima TXT i PNG
"""

import csv
import math
import os
import time
from datetime import timedelta

import matplotlib.pyplot as plt
import numpy as np


T0 = time.time()

CSV_DRAWS = "/data/loto7_4624_k43.csv"

HERE = os.path.dirname(os.path.abspath(__file__))
TXT_PATH = os.path.join(HERE, "MonteCarlo_1_v2.txt")
PNG_PATH = os.path.join(HERE, "MonteCarlo_1_v2.png")

N_MAX = 39
K_PICK = 7
TOTAL_COMBOS = math.comb(N_MAX, K_PICK)

MC_SEED = 39
MC_RUNS = 1_000_000
BATCH_SIZE = 100_000
REF_RUNS = 50_000
TOP_KEEP = 20
TOP_POOL_KEEP = 2_500
DIVERSITY_MAX_OVERLAP = 4
RECENT_WINDOW = 250


def read_loto_csv(path):
    rows = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < K_PICK:
                continue
            try:
                nums = tuple(sorted(int(x) for x in row[:K_PICK]))
            except ValueError:
                continue
            if (
                len(nums) == K_PICK
                and len(set(nums)) == K_PICK
                and all(1 <= x <= N_MAX for x in nums)
            ):
                rows.append(nums)
    return rows


def random_combos_fast(rng, n_rows):
    """Uniformno uzorkovanje 7/39 kombinacija preko random kljuceva."""
    keys = rng.random((n_rows, N_MAX))
    picks = np.argpartition(keys, K_PICK - 1, axis=1)[:, :K_PICK] + 1
    picks.sort(axis=1)
    return picks.astype(np.int16)


def combo_features(combos):
    combos = np.asarray(combos, dtype=np.int16)
    sums = combos.sum(axis=1)
    odd_counts = (combos % 2 == 1).sum(axis=1)
    high_counts = (combos >= 20).sum(axis=1)
    consecutive_pairs = (np.diff(combos, axis=1) == 1).sum(axis=1)
    return sums, odd_counts, high_counts, consecutive_pairs


def combo_masks(combos):
    masks = np.zeros((len(combos), N_MAX + 1), dtype=np.uint8)
    rows = np.arange(len(combos))[:, None]
    masks[rows, combos] = 1
    return masks


def max_overlap_recent(combos, recent_masks, chunk=20_000):
    masks = combo_masks(combos)
    out = np.empty(len(combos), dtype=np.int16)
    for start in range(0, len(combos), chunk):
        end = min(start + chunk, len(combos))
        overlaps = masks[start:end] @ recent_masks.T
        out[start:end] = overlaps.max(axis=1)
    return out


def build_empirical_model(train):
    sums, odds, highs, consec = combo_features(train)
    min_sum = K_PICK * (K_PICK + 1) // 2
    max_sum = sum(range(N_MAX - K_PICK + 1, N_MAX + 1))

    alpha = 1.0
    sum_counts = np.bincount(sums, minlength=max_sum + 1).astype(float)
    sum_probs = sum_counts + alpha
    sum_probs[:min_sum] = 0.0
    sum_probs = sum_probs / sum_probs.sum()

    odd_probs = np.bincount(odds, minlength=K_PICK + 1).astype(float) + alpha
    odd_probs = odd_probs / odd_probs.sum()

    high_probs = np.bincount(highs, minlength=K_PICK + 1).astype(float) + alpha
    high_probs = high_probs / high_probs.sum()

    consec_probs = np.bincount(consec, minlength=K_PICK).astype(float) + alpha
    consec_probs = consec_probs / consec_probs.sum()

    gap_counts = np.ones(N_MAX + 1, dtype=float) * alpha
    gaps = np.diff(train, axis=1)
    for g in gaps.ravel():
        gap_counts[int(g)] += 1.0
    gap_probs = gap_counts / gap_counts.sum()

    pair_counts = np.ones((N_MAX + 1, N_MAX + 1), dtype=float) * alpha
    for row in train:
        for i in range(K_PICK):
            for j in range(i + 1, K_PICK):
                a, b = int(row[i]), int(row[j])
                pair_counts[a, b] += 1.0
                pair_counts[b, a] += 1.0
    pair_probs = pair_counts / pair_counts.sum()

    recent = train[-min(RECENT_WINDOW, len(train)):]
    return {
        "log_sum": np.log(sum_probs + 1e-300),
        "log_odd": np.log(odd_probs + 1e-300),
        "log_high": np.log(high_probs + 1e-300),
        "log_consec": np.log(consec_probs + 1e-300),
        "log_gap": np.log(gap_probs + 1e-300),
        "log_pair": np.log(pair_probs + 1e-300),
        "recent_masks": combo_masks(recent),
        "sum_q": np.quantile(sums, [0.25, 0.50, 0.75]),
    }


def raw_components(combos, model):
    combos = np.asarray(combos, dtype=np.int16)
    sums, odds, highs, consec = combo_features(combos)

    shape = (
        model["log_sum"][sums]
        + model["log_odd"][odds]
        + model["log_high"][highs]
        + model["log_consec"][consec]
    )

    gaps = np.diff(combos, axis=1)
    gap_score = model["log_gap"][gaps].mean(axis=1)

    pair_score = np.zeros(len(combos), dtype=float)
    pair_n = 0
    for i in range(K_PICK):
        for j in range(i + 1, K_PICK):
            pair_score += model["log_pair"][combos[:, i], combos[:, j]]
            pair_n += 1
    pair_score = pair_score / pair_n

    max_ov = max_overlap_recent(combos, model["recent_masks"])
    novelty = -max_ov.astype(float)

    return np.column_stack([shape, gap_score, pair_score, novelty])


def standardize_components(raw, ref_mean, ref_std):
    return (raw - ref_mean) / (ref_std + 1e-12)


def choose_weights(train, valid, rng):
    model = build_empirical_model(train)
    ref = random_combos_fast(rng, REF_RUNS)

    ref_raw = raw_components(ref, model)
    ref_mean = ref_raw.mean(axis=0)
    ref_std = ref_raw.std(axis=0, ddof=1)
    ref_z = standardize_components(ref_raw, ref_mean, ref_std)
    valid_z = standardize_components(raw_components(valid, model), ref_mean, ref_std)

    candidates = [
        (0.30, 0.20, 0.40, 0.10),
        (0.25, 0.25, 0.40, 0.10),
        (0.20, 0.25, 0.45, 0.10),
        (0.20, 0.30, 0.40, 0.10),
        (0.25, 0.20, 0.45, 0.10),
        (0.25, 0.15, 0.50, 0.10),
        (0.30, 0.15, 0.45, 0.10),
        (0.20, 0.20, 0.45, 0.15),
        (0.25, 0.20, 0.40, 0.15),
        (0.30, 0.20, 0.35, 0.15),
    ]

    rows = []
    for weights in candidates:
        w = np.asarray(weights, dtype=float)
        ref_score = ref_z @ w
        valid_score = valid_z @ w
        percentiles = [float(np.mean(ref_score <= s)) for s in valid_score]
        metric = float(np.mean(percentiles))
        rows.append((metric, weights))

    rows.sort(key=lambda x: x[0], reverse=True)
    best_metric, best_weights = rows[0]
    return best_weights, best_metric, rows, model, ref_mean, ref_std


def diversify_top(pool_combos, pool_scores, top_keep=TOP_KEEP):
    order = np.argsort(pool_scores)[::-1]
    selected = []
    selected_scores = []
    selected_sets = []
    seen = set()

    for idx in order:
        combo = tuple(int(x) for x in pool_combos[idx])
        if combo in seen:
            continue
        cset = set(combo)
        if any(len(cset & old) > DIVERSITY_MAX_OVERLAP for old in selected_sets):
            continue
        selected.append(combo)
        selected_scores.append(float(pool_scores[idx]))
        selected_sets.append(cset)
        seen.add(combo)
        if len(selected) >= top_keep:
            break

    return list(zip(selected, selected_scores))


draws = read_loto_csv(CSV_DRAWS)
hist = np.asarray(draws, dtype=np.int16)
N = len(hist)
split = int(N * 0.90)
train = hist[:split]
valid = hist[split:]

rng = np.random.default_rng(MC_SEED)
best_weights, best_metric, weight_rows, model, ref_mean, ref_std = choose_weights(
    train, valid, rng
)
weights = np.asarray(best_weights, dtype=float)

pool_combos = np.empty((0, K_PICK), dtype=np.int16)
pool_scores = np.empty(0, dtype=float)

done = 0
while done < MC_RUNS:
    n_batch = min(BATCH_SIZE, MC_RUNS - done)
    batch = random_combos_fast(rng, n_batch)
    comps = standardize_components(raw_components(batch, model), ref_mean, ref_std)
    scores = comps @ weights

    pool_combos = np.vstack([pool_combos, batch])
    pool_scores = np.concatenate([pool_scores, scores])
    if len(pool_scores) > TOP_POOL_KEEP:
        keep_idx = np.argsort(pool_scores)[-TOP_POOL_KEEP:]
        pool_combos = pool_combos[keep_idx]
        pool_scores = pool_scores[keep_idx]
    done += n_batch

top_rows = diversify_top(pool_combos, pool_scores, TOP_KEEP)

sums_hist, odds_hist, highs_hist, consec_hist = combo_features(hist)
ref_plot = random_combos_fast(rng, REF_RUNS)
sums_ref, odds_ref, highs_ref, consec_ref = combo_features(ref_plot)

sum_q25, sum_q50, sum_q75 = model["sum_q"]
prob_sum_iqr = float(np.mean((sums_ref >= sum_q25) & (sums_ref <= sum_q75)))
prob_shape_close = float(np.mean(
    (np.abs(odds_ref - round(odds_hist.mean())) <= 1)
    & (np.abs(highs_ref - round(highs_hist.mean())) <= 1)
    & (np.abs(consec_ref - round(consec_hist.mean())) <= 1)
))

lines = []
lines.append("MonteCarlo_1_v2 — jaca MC optimizacija bez proste frekvencije brojeva")
lines.append("=" * 72)
lines.append("")
lines.append(f"CSV:                  {CSV_DRAWS}")
lines.append(f"Ucitano izvlacenja:    {N}")
lines.append(f"Train/valid split:     {len(train)} / {len(valid)}")
lines.append(f"C(39,7):              {TOTAL_COMBOS:,}")
lines.append(f"MC_RUNS:              {MC_RUNS:,}")
lines.append(f"BATCH_SIZE:           {BATCH_SIZE:,}")
lines.append(f"MC_SEED:              {MC_SEED}")
lines.append("")
lines.append("Skor komponente (bez proste frekvencije pojedinacnih brojeva):")
lines.append("  shape  = logP(sum) + logP(odd) + logP(high) + logP(consecutive)")
lines.append("  gap    = prosecan logP(razmaka izmedju susednih brojeva)")
lines.append("  pair   = prosecan logP(parova brojeva u kombinaciji)")
lines.append("  novelty= penal za preveliko preklapanje sa skorijom istorijom")
lines.append("")
lines.append("Mini-backtest tezine:")
lines.append(f"  best weights (shape, gap, pair, novelty) = {best_weights}")
lines.append(f"  mean validation percentile vs MC reference = {best_metric:.6f}")
lines.append("")
lines.append("Top tezine iz mini-backtesta:")
for metric, row in weight_rows[:5]:
    lines.append(f"  metric={metric:.6f}  weights={row}")
lines.append("")
lines.append("MC numericka integracija / referenca:")
lines.append(f"  P(sum u istorijskom IQR [{sum_q25:.0f}, {sum_q75:.0f}]) = {prob_sum_iqr:.6f}")
lines.append(f"  P(odd/high/consecutive blizu istorijskog oblika)        = {prob_shape_close:.6f}")
lines.append("")
lines.append("Istorijske statistike oblika:")
lines.append(f"  sum:    mean={sums_hist.mean():.2f} q25={sum_q25:.0f} median={sum_q50:.0f} q75={sum_q75:.0f}")
lines.append(f"  odd:    mean={odds_hist.mean():.4f}")
lines.append(f"  high:   mean={highs_hist.mean():.4f}")
lines.append(f"  consec: mean={consec_hist.mean():.4f}")
lines.append("")
lines.append(f"Monte Carlo optimizacija — top {len(top_rows)} diverzifikovanih kandidata:")
lines.append(f"  {'rank':>4}  {'kombinacija':<32}{'score':>12}")
for rank, (combo, score) in enumerate(top_rows, start=1):
    lines.append(f"  {rank:>4}  {str(combo):<32}{score:>12.6f}")

elapsed = time.time() - T0
lines.append("")
lines.append(f"Ukupno vreme: {timedelta(seconds=int(elapsed))} ({elapsed:.1f} s)")

text = "\n".join(lines) + "\n"
print(text)
with open(TXT_PATH, "w", encoding="utf-8") as f:
    f.write(text)


fig, ax = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle("MonteCarlo_1_v2 — skor bez proste frekvencije brojeva",
             fontsize=14, fontweight="bold")

bins_sum = np.arange(28, 252, 6)
ax[0, 0].hist(sums_ref, bins=bins_sum, density=True, color="lightgray",
              edgecolor="white", label="MC uniform")
ax[0, 0].hist(sums_hist, bins=bins_sum, density=True, color="darkorange",
              alpha=0.55, edgecolor="white", label="istorija")
ax[0, 0].axvline(sum_q25, color="black", linestyle="--", linewidth=1)
ax[0, 0].axvline(sum_q75, color="black", linestyle="--", linewidth=1)
ax[0, 0].set_title("Distribucija sume kombinacije")
ax[0, 0].set_xlabel("sum")
ax[0, 0].set_ylabel("density")
ax[0, 0].legend(fontsize=8)
ax[0, 0].grid(True, alpha=0.2)

metrics = [row[0] for row in weight_rows]
labels = [str(row[1]) for row in weight_rows]
ax[0, 1].barh(np.arange(len(metrics)), metrics, color="steelblue", alpha=0.85)
ax[0, 1].set_yticks(np.arange(len(metrics)))
ax[0, 1].set_yticklabels(labels, fontsize=7)
ax[0, 1].invert_yaxis()
ax[0, 1].set_title("Mini-backtest: percentile metric")
ax[0, 1].set_xlabel("mean validation percentile")
ax[0, 1].grid(True, alpha=0.2, axis="x")

pair_log = model["log_pair"][1:, 1:]
im = ax[1, 0].imshow(pair_log, cmap="viridis", aspect="auto")
ax[1, 0].set_title("Log-verovatnoce parova (train)")
ax[1, 0].set_xlabel("broj")
ax[1, 0].set_ylabel("broj")
ax[1, 0].set_xticks(np.arange(0, N_MAX, 5))
ax[1, 0].set_xticklabels(np.arange(1, N_MAX + 1, 5))
ax[1, 0].set_yticks(np.arange(0, N_MAX, 5))
ax[1, 0].set_yticklabels(np.arange(1, N_MAX + 1, 5))
fig.colorbar(im, ax=ax[1, 0], fraction=0.046, pad=0.04)

top_show = top_rows[:10]
combo_labels = ["-".join(str(x) for x in combo) for combo, _ in top_show]
scores = [score for _, score in top_show]
ax[1, 1].barh(np.arange(len(scores)), scores, color="seagreen", alpha=0.85)
ax[1, 1].set_yticks(np.arange(len(scores)))
ax[1, 1].set_yticklabels(combo_labels, fontsize=8)
ax[1, 1].invert_yaxis()
ax[1, 1].set_title("Top diverzifikovani MC kandidati")
ax[1, 1].set_xlabel("score")
ax[1, 1].grid(True, alpha=0.2, axis="x")

for a in ax.ravel():
    a.spines["top"].set_visible(False)
    a.spines["right"].set_visible(False)

fig.tight_layout()
plt.show()
fig.savefig(PNG_PATH, dpi=160, bbox_inches="tight")

print(f"TXT saved -> {TXT_PATH}")
print(f"PNG saved -> {PNG_PATH}")
print()



"""
MonteCarlo_1_v2 — jaca MC optimizacija bez proste frekvencije brojeva
========================================================================

CSV:                  /data/loto7_4624_k43.csv
Ucitano izvlacenja:   4624
Train/valid split:    4161 / 463
C(39,7):              15,380,937
MC_RUNS:              1,000,000
BATCH_SIZE:           100,000
MC_SEED:              39

Skor komponente (bez proste frekvencije pojedinacnih brojeva):
  shape  = logP(sum) + logP(odd) + logP(high) + logP(consecutive)
  gap    = prosecan logP(razmaka izmedju susednih brojeva)
  pair   = prosecan logP(parova brojeva u kombinaciji)
  novelty= penal za preveliko preklapanje sa skorijom istorijom

Mini-backtest tezine:
  best weights (shape, gap, pair, novelty) = (0.25, 0.15, 0.5, 0.1)
  mean validation percentile vs MC reference = 0.510540

Top tezine iz mini-backtesta:
  metric=0.510540  weights=(0.25, 0.15, 0.5, 0.1)
  metric=0.510258  weights=(0.3, 0.15, 0.45, 0.1)
  metric=0.508798  weights=(0.25, 0.2, 0.45, 0.1)
  metric=0.508539  weights=(0.3, 0.2, 0.4, 0.1)
  metric=0.507772  weights=(0.25, 0.2, 0.4, 0.15)

MC numericka integracija / referenca:
  P(sum u istorijskom IQR [121, 160]) = 0.524320
  P(odd/high/consecutive blizu istorijskog oblika)        = 0.551840

Istorijske statistike oblika:
  sum:    mean=140.51 q25=121 median=140 q75=160
  odd:    mean=3.5926
  high:   mean=3.6157
  consec: mean=1.0692

Monte Carlo optimizacija — top 20 diverzifikovanih kandidata:
  rank  kombinacija                            score
     1  (8, x, 14, y, 26, z, 34)         2.608757
     2  (8, x, 16, y, 23, z, 37)         2.405372
     3  (8, x, 23, y, 26, z, 35)         2.390343
     4  (8, x, 22, y, 26, z, 34)         2.331274
     5  (7, x, 11, y, 26, z, 37)          2.307115
     6  (5, x, 11, y, 23, z, 28)          2.248418
     7  (7, x, 18, y, 26, z, 35)          2.238544
     8  (7, x, 13, y, 24, z, 33)          2.204225
     9  (7, x, 10, y, 23, z, 26)          2.143112
    10  (8, x, 16, y, 23, z, 26)         2.091236
    11  (8, x, 18, y, 22, z, 25)         2.081555
    12  (4, x, 8, y, 26, z, 38)           2.075488
    13  (8, x, 11, y, 29, z, 37)         2.068431
    14  (4, x, 9, y, 22, z, 26)           2.067414
    15  (8, x, 10, y, 23, z, 34)          2.052232
    16  (8, x, 11, y, 28, z, 33)          2.050135
    17  (8, x, 11, y, 28, z, 34)         2.045820
    18  (8, x, 11, y, 25, z, 34)          2.005132
    19  (8, x, 13, y, 25, z, 35)         1.996984
    20  (8, x, 11, y, 26, z, 34)          1.985844

Ukupno vreme: 0:00:06 (6.6 s)

TXT saved -> /MonteCarlo_1_v2.txt
PNG saved -> /MonteCarlo_1_v2.png
"""



"""
Skor bez proste frekvencije brojeva. Umesto „hot count"-a koristim:
razlike u distribuciji suma/odd/high/consecutive (KS distanca prema istoriji),
distribucija razmaka (gap-ovi između susednih brojeva) prema istoriji,
pair co-occurrence matrica (parovi brojeva koji se često zajedno javljaju, bez direktnog brojanja pojedinačnih brojeva).
Težine skora optimizovane mini-backtestom (split istorije, traži težine koje najbolje prate poslednji ~10% kola).
Seed=39, MC_RUNS=1_000_000 (umesto 100k), batch generisanje radi brzine.
Diverzifikacija top-N: bira TOP_KEEP kandidata uz uslov da nijedan par ne deli više od D istih brojeva (npr. D=4), da lista ne bude monotona.
TXT/PNG izlazi analogni MonteCarlo_1, sa dodatkom backtest grafa i KS distanci.

Fokus: seed 39, bez proste frekvencije brojeva, skor preko parova, oblika kombinacije, gapova i mini-backtest podešavanja težina.

MC_SEED = 39
MC_RUNS = 1_000_000
brzo uniformno uzorkovanje 7/39 preko random ključeva
skor bez proste frekvencije pojedinačnih brojeva
komponente skora:
oblik kombinacije: suma, odd, high, consecutive
gap distribucija
pair co-occurrence matrica
novelty penal za preveliko preklapanje sa skorijom istorijom
mini-backtest za izbor težina skora
diverzifikaciju top kandidata
izlaze:
MonteCarlo_1_v2.txt
MonteCarlo_1_v2.png
"""
