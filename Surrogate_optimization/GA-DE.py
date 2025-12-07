# -*- coding: utf-8 -*-
"""
GA + DE hybrid optimization (maximize Fy/G; FT-Transformer surrogate)
======================================================================
# ========= Hybrid GA–DE Hyperparameters (Table \ref{tab:hyperparams}) =========
POP_SIZE           = 100            # N_pop: population size
N_GENERATIONS      = 80             # G_max: maximum generations

CROSS_P            = 0.90           # p_c: GA crossover probability
MUT_P              = 0.45           # p_m: GA mutation probability

DE_PORTION         = 0.25           # p_DE: fraction of population evolved by DE
DE_PBEST_FRAC      = 0.20           # p_pbest: fraction of top individuals as p-best

DE_F_RANGE         = (0.25, 0.65)   # F in [0.25, 0.65]
DE_CR_RANGE        = (0.30, 0.60)   # CR in [0.3, 0.6]

ARCHIVE_MAX        = 100            # N_arch: maximum archive size

EPSILON_START      = 6.0            # ε_0: initial ε-relaxation margin (N)
EPSILON_DECAY_GENS = 12             # G_a: ε-annealing duration (generations)

REPAIR_DELTA_TH    = 5.0            # δ_th: max violation for applying local repair (N)
"""

import os, math, time, random, csv, datetime
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# ========= Base configuration =========
save_dir      = r'runs/ft_transformer'   # FT-Transformer model directory
weights_name  = r'model.pt'
activation    = 'gelu'
use_log_ratio = False

VAR_BOUNDS = {'d': (3.0, 20.0), 'w': (10.0, 35.0), 'n1': (0.33, 3.0), 'R': (20.0, 30.0)}

g = 9.81
m0 = 2.0
G0 = m0 * g
mu = 0.35
h_mm = 75.0
R1_mm = 100.0
L1_mm = 67.23
L2_mm = 39.56
L3_mm = 29.63
KAPPA = 2.0
SAFETY_FACTOR = 1.5

POP_SIZE = 100
N_GENERATIONS = 80
TOURN_K = 2
CROSS_P = 0.9
MUT_P = 0.45
MUT_SIGMA = {'d': 1.5, 'w': 1.5, 'n1': 0.08, 'R': 1.2}
ELITE_KEEP = 2

DE_PORTION = 0.25
DE_F_RANGE = (0.25, 0.65)
DE_CR_RANGE = (0.3, 0.6)
DE_PBEST_FRAC = 0.2
ARCHIVE_MAX = 100

INIT_LHS_PORTION = 0.75
INIT_EDGE_PORTION = 0.20
INIT_FEASIBLE_TARGET = 0.20
INIT_RESAMPLE_PASSES = 1

ENABLE_EPSILON = True
EPSILON_START = 6.0
EPSILON_DECAY_GENS = 12


def epsilon_at_gen(gen: int) -> float:
    if not ENABLE_EPSILON:
        return 0.0
    t = max(0, min(gen - 1, EPSILON_DECAY_GENS))
    return EPSILON_START * (1.0 - t / float(EPSILON_DECAY_GENS))


ENABLE_REPAIR = True
REPAIR_DELTA_TH = 5.0
REPAIR_MAX_BISECT = 10
REPAIR_COORD_STEPS = 3
REPAIR_STEP = {'R': 0.15, 'd': 0.4, 'w': 0.4, 'n1': 0.01}

BASE_NAME = "GADE-FTTransformer"
SEED = int(time.time()) % (2**32)
PRINT_EVERY = 1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPS = 1e-8

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if device.type == 'cuda':
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

ts = datetime.datetime.now().strftime("%m%d%H%M%S")
OUTDIR = f"{BASE_NAME}-{ts}"
os.makedirs(OUTDIR, exist_ok=True)
HIST_CSV = os.path.join(OUTDIR, "ga_history.csv")
VIO_STATS_CSV = os.path.join(OUTDIR, "ga_violation_stats.csv")
VIO_SAMPLES_CSV = os.path.join(OUTDIR, "ga_violation_samples.csv")
DIV_CSV = os.path.join(OUTDIR, "ga_diversity.csv")

# ========= FT-Transformer model =========
class Tokenizer(nn.Module):
    def __init__(self, d_numerical, d_token, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(d_numerical + 1, d_token))
        self.bias = nn.Parameter(torch.empty(d_numerical, d_token)) if bias else None
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            nn.init.kaiming_uniform_(self.bias, a=math.sqrt(5))

    def forward(self, x_num):
        x_num = torch.cat([torch.ones(len(x_num), 1, device=x_num.device), x_num], dim=1)
        x = self.weight[None] * x_num[:, :, None]
        if self.bias is not None:
            bias = torch.cat([torch.zeros(1, self.bias.shape[1], device=x.device), self.bias])
            x = x + bias[None]
        return x


class MultiheadAttention(nn.Module):
    def __init__(self, d, n_heads, dropout):
        super().__init__()
        self.n_heads = n_heads
        self.W_q, self.W_k, self.W_v = nn.Linear(d, d), nn.Linear(d, d), nn.Linear(d, d)
        self.W_out = nn.Linear(d, d)
        self.dropout = nn.Dropout(dropout)

    def _reshape(self, x):
        B, N, D = x.shape
        d_h = D // self.n_heads
        return x.reshape(B, N, self.n_heads, d_h).transpose(1, 2).reshape(B * self.n_heads, N, d_h)

    def forward(self, q, kv):
        q, k, v = self.W_q(q), self.W_k(kv), self.W_v(kv)
        q, k, v = self._reshape(q), self._reshape(k), self._reshape(v)
        att = torch.softmax(q @ k.transpose(1, 2) / math.sqrt(k.shape[-1]), dim=-1)
        att = self.dropout(att)
        x = att @ v
        B = len(q) // self.n_heads
        x = x.reshape(B, self.n_heads, -1, v.shape[-1]).transpose(1, 2).reshape(
            B, -1, self.n_heads * v.shape[-1]
        )
        return self.W_out(x)


class FTTransformer(nn.Module):
    def __init__(
        self,
        d_numerical=4,
        n_layers=4,
        d_token=128,
        n_heads=4,
        d_ffn_factor=4.0,
        attention_dropout=0.1,
        ffn_dropout=0.1,
        activation="gelu",
        d_out=3,
    ):
        super().__init__()
        self.tokenizer = Tokenizer(d_numerical, d_token, True)
        self.layers = nn.ModuleList([])
        self.act = torch.nn.functional.gelu if activation == "gelu" else torch.relu
        d_hidden = int(d_token * d_ffn_factor)
        for _ in range(n_layers):
            self.layers.append(
                nn.ModuleDict(
                    {
                        "attn": MultiheadAttention(d_token, n_heads, attention_dropout),
                        "norm0": nn.LayerNorm(d_token),
                        "norm1": nn.LayerNorm(d_token),
                        "fc1": nn.Linear(d_token, d_hidden),
                        "fc2": nn.Linear(d_hidden, d_token),
                    }
                )
            )
        self.head = nn.Linear(d_token, d_out)

    def forward(self, x_num):
        x = self.tokenizer(x_num)
        for l in self.layers:
            x = x + l["attn"](l["norm0"](x), l["norm0"](x))
            ff = self.act(l["fc1"](l["norm1"](x)))
            x = x + l["fc2"](ff)
        return self.head(torch.nn.functional.gelu(x[:, 0]))


# ========= Model loading and prediction =========
def load_scalers(dirpath):
    sx = joblib.load(os.path.join(dirpath, "x_scaler.joblib"))
    sy = joblib.load(os.path.join(dirpath, "y_scaler.joblib"))
    return sx, sy


def load_model(dirpath, weights_name, activation):
    model = FTTransformer(
        d_numerical=4,
        n_layers=4,
        d_token=128,
        n_heads=4,
        d_ffn_factor=4.0,
        attention_dropout=0.1,
        ffn_dropout=0.1,
        activation=activation,
        d_out=3,
    ).to(device)
    state = torch.load(os.path.join(dirpath, weights_name), map_location=device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


@torch.no_grad()
def model_predict(model, scaler_X, scaler_Y, xs, use_log_ratio=False):
    x_scaled = scaler_X.transform(xs.astype(np.float32))
    y_scaled = model(torch.tensor(x_scaled, dtype=torch.float32, device=device)).cpu().numpy()
    y = scaler_Y.inverse_transform(y_scaled)
    Fy, G, ratio = y[:, 0], y[:, 1], y[:, 2]
    if use_log_ratio:
        ratio = np.exp(ratio)
    return Fy, G, ratio


# ========= Mechanics, GA/DE, repair, visualization =========
def solve_alpha_for_R(R_mm):
    A = R1_mm + 2.0 * R_mm + L3_mm
    B = L2_mm
    C = L1_mm / 2.0
    Rmag = math.hypot(A, B)
    if Rmag < 1e-9:
        return 0.0, C
    s = max(-1.0, min(1.0, C / Rmag))
    phi = math.atan2(B, A)
    asin_s = math.asin(s)
    cand = [phi + asin_s, phi + (math.pi - asin_s)]
    best_a, best_r = None, None
    for a in cand:
        a2 = min(max(a, 0.0), math.pi / 2)
        resid = C - (A * math.sin(a2) - B * math.cos(a2))
        if (best_r is None) or (abs(resid) < abs(best_r)):
            best_a, best_r = a2, resid
    return float(best_a), float(best_r)


def required_Fy_min(G, alpha):
    G = np.asarray(G, dtype=np.float64)
    alpha = np.asarray(alpha, dtype=np.float64)
    threeG_plus_G0 = 3.0 * G + G0
    k1 = math.hypot(1.0, (1.0 + h_mm / R1_mm)) / (3.0 * mu)
    term1 = threeG_plus_G0 * k1
    denom = 1.0 + KAPPA * (np.cos(alpha) - mu * np.sin(alpha))
    term2 = np.where(denom > 0.0, threeG_plus_G0 / np.maximum(denom, 1e-6), -1e12)
    return np.maximum(term1, term2)


# ========= Operators & utilities =========
GENES = ['d', 'w', 'n1', 'R']


def clamp_ind(ind):
    for i, gk in enumerate(GENES):
        lo, hi = VAR_BOUNDS[gk]
        ind[i] = min(max(ind[i], lo), hi)
    return ind


def uniform_crossover(p1, p2, prob=CROSS_P):
    if np.random.rand() > prob:
        return p1.copy(), p2.copy()
    m = np.random.rand(len(p1)) < 0.5
    c1, c2 = p1.copy(), p2.copy()
    c1[m], c2[m] = p2[m], p1[m]
    return c1, c2


def mutate(ind, prob=MUT_P):
    for i, gk in enumerate(GENES):
        if np.random.rand() < prob:
            ind[i] += np.random.normal(0.0, MUT_SIGMA.get(gk, 0.1))
    return clamp_ind(ind)


# Tournament selection with Deb's rule
def tournament_select(pop, fits, k=TOURN_K, Fy=None, Fy_need=None, eps=0.0):
    idxs = np.random.choice(len(pop), size=k, replace=False)

    def key(j):
        if (Fy is not None) and (Fy_need is not None):
            feasible = (fits[j] > -1e-11) and (Fy[j] >= SAFETY_FACTOR * Fy_need[j] - eps)
            if feasible:
                return (2, fits[j])
            gap = SAFETY_FACTOR * Fy_need[j] - Fy[j]
            return (1, -gap)
        return (0, fits[j])

    best = max(idxs, key=key)
    return pop[best].copy()


# Single-individual fitness (with epsilon)
def fitness_one(ind, model, scaler_X, scaler_Y, eps=0.0):
    xs = np.array([ind], dtype=np.float32)
    Fy, G, _ = model_predict(model, scaler_X, scaler_Y, xs, use_log_ratio=use_log_ratio)
    Fy, G = float(Fy[0]), float(G[0])
    alpha_i, _ = solve_alpha_for_R(float(ind[3]))
    Fy_need = float(required_Fy_min(np.array([G]), np.array([alpha_i]))[0])
    fit = -1e12
    if (G > 0.0) and (Fy >= SAFETY_FACTOR * Fy_need - eps):
        fit = Fy / max(G, EPS)
    return fit, Fy, G, Fy_need, alpha_i


# Population fitness (with epsilon)
def population_fitness(pop, model, scaler_X, scaler_Y, eps=0.0):
    xs = np.stack(pop, axis=0).astype(np.float32)
    Fy, G, _ = model_predict(model, scaler_X, scaler_Y, xs, use_log_ratio=use_log_ratio)
    R_all = xs[:, 3].astype(np.float64)
    alpha_all = np.array([solve_alpha_for_R(float(Ri))[0] for Ri in R_all], dtype=np.float64)
    Fy_need = required_Fy_min(G, alpha_all)
    fits = np.full(len(pop), -1e12, dtype=np.float64)
    ok = (G > 0.0) & (Fy >= SAFETY_FACTOR * Fy_need - eps)
    fits[ok] = Fy[ok] / np.maximum(G[ok], EPS)
    return fits, Fy, G, Fy_need, alpha_all


# ========= Feasibility repair =========
def violation_delta_for(ind, model, scaler_X, scaler_Y):
    fit, Fy, G, Fy_need, _ = fitness_one(ind, model, scaler_X, scaler_Y, eps=0.0)
    delta = SAFETY_FACTOR * Fy_need - Fy
    return float(delta), float(Fy), float(G)


def feasibility_repair(child, parent, model, scaler_X, scaler_Y, delta_th=REPAIR_DELTA_TH):
    if not ENABLE_REPAIR:
        return child
    delta_c, _, _ = violation_delta_for(child, model, scaler_X, scaler_Y)
    if not (0.0 < delta_c <= delta_th):
        return child

    delta_p, _, _ = violation_delta_for(parent, model, scaler_X, scaler_Y)
    parent_feasible = delta_p <= 1e-12
    if parent_feasible:
        t_lo, t_hi = 0.0, 1.0
        found = False
        for _ in range(REPAIR_MAX_BISECT):
            tm = 0.5 * (t_lo + t_hi)
            xm = (1.0 - tm) * parent + tm * child
            clamp_ind(xm)
            delta_m, _, _ = violation_delta_for(xm, model, scaler_X, scaler_Y)
            if delta_m <= 1e-12:
                found = True
                t_lo = tm
            else:
                t_hi = tm
        if found:
            x_line = (1.0 - t_lo) * parent + t_lo * child
            clamp_ind(x_line)
            return x_line

    order = ['R', 'd', 'w', 'n1']
    x_try = child.copy()
    best_delta = delta_c
    for _ in range(REPAIR_COORD_STEPS):
        improved = False
        for key in order:
            j = GENES.index(key)
            step = REPAIR_STEP.get(key, 0.1)
            for sgn in (+1, -1):
                y = x_try.copy()
                y[j] += sgn * step
                clamp_ind(y)
                delta_y, _, _ = violation_delta_for(y, model, scaler_X, scaler_Y)
                if delta_y < best_delta - 1e-9:
                    best_delta = delta_y
                    x_try = y
                    improved = True
                    if best_delta <= 1e-12:
                        return x_try
        if not improved:
            break
    return x_try


# ========= Initialization =========
def lhs_sample(n, dim):
    seg = np.linspace(0.0, 1.0, n + 1)
    pts = np.zeros((n, dim), dtype=np.float64)
    for j in range(dim):
        u = np.random.rand(n) * (seg[1:] - seg[:-1]) + seg[:-1]
        np.random.shuffle(u)
        pts[:, j] = u
    return pts


def scale_to_bounds(U):
    arr = np.zeros_like(U, dtype=np.float64)
    for j, gk in enumerate(GENES):
        lo, hi = VAR_BOUNDS[gk]
        arr[:, j] = lo + U[:, j] * (hi - lo)
    return arr


def edge_samples(m):
    per_dim = max(1, m // (2 * len(GENES)))
    samples = []
    for j, gk in enumerate(GENES):
        lo, hi = VAR_BOUNDS[gk]
        for side in [lo, hi]:
            for _ in range(per_dim):
                ind = np.array(
                    [np.random.uniform(*VAR_BOUNDS[gg]) for gg in GENES],
                    dtype=np.float64,
                )
                ind[j] = side
                samples.append(clamp_ind(ind))
    while len(samples) < m:
        ind = np.array(
            [np.random.uniform(*VAR_BOUNDS[gg]) for gg in GENES],
            dtype=np.float64,
        )
        samples.append(clamp_ind(ind))
    return np.array(samples[:m], dtype=np.float64)


def init_population(model, scaler_X, scaler_Y, pop_size=POP_SIZE):
    n_lhs = int(round(pop_size * INIT_LHS_PORTION))
    n_edge = int(round(pop_size * INIT_EDGE_PORTION))
    n_rand = max(0, pop_size - n_lhs - n_edge)
    U = lhs_sample(n_lhs, len(GENES))
    lhs = scale_to_bounds(U)
    edge = edge_samples(n_edge)
    rnd = np.array(
        [[np.random.uniform(*VAR_BOUNDS[gk]) for gk in GENES] for _ in range(n_rand)],
        dtype=np.float64,
    )
    pop = np.vstack([lhs, edge, rnd])
    pop = [clamp_ind(ind) for ind in pop]

    for _ in range(INIT_RESAMPLE_PASSES):
        fits, _, _, _, _ = population_fitness(pop, model, scaler_X, scaler_Y)
        feasible_ratio = float(np.mean(fits > -1e11)) if len(pop) > 0 else 0.0
        if feasible_ratio >= INIT_FEASIBLE_TARGET:
            break
        for i, ok in enumerate(fits > -1e11):
            if not ok:
                pop[i] = np.array(
                    [np.random.uniform(*VAR_BOUNDS[gk]) for gk in GENES],
                    dtype=np.float64,
                )
                clamp_ind(pop[i])
    return pop


# ========= DE step: current-to-pbest/1 with external archive =========
def de_step_inplace_pbest(
    pop,
    fits,
    model,
    scaler_X,
    scaler_Y,
    archive,
    portion=DE_PORTION,
    f_range=DE_F_RANGE,
    cr_range=DE_CR_RANGE,
    pbest_frac=DE_PBEST_FRAC,
    eps=0.0,
):
    N = len(pop)
    m = max(1, int(round(N * portion)))
    idx_targets = np.random.choice(N, size=m, replace=False)

    k = max(2, int(math.ceil(pbest_frac * N)))
    elite_idx = np.argsort(fits)[-k:]
    elite_idx = elite_idx[~np.isnan(fits[elite_idx])]

    for idx in idx_targets:
        F = np.random.uniform(*f_range)
        CR = np.random.uniform(*cr_range)

        x_i = pop[idx]
        if elite_idx.size == 0:
            pbest_idx = int(np.argmax(fits))
        else:
            pbest_idx = int(np.random.choice(elite_idx))
        x_pbest = pop[pbest_idx]

        candidates = list(set(range(N)) - {idx, pbest_idx})
        if len(candidates) < 2:
            continue
        r1 = int(np.random.choice(candidates))
        x_r1 = pop[r1]

        pool_vectors = []
        for j in range(N):
            if j not in {idx, pbest_idx, r1}:
                pool_vectors.append(pop[j])
        if archive:
            pool_vectors.extend(archive)
        if len(pool_vectors) == 0:
            continue
        x_r2 = pool_vectors[np.random.randint(len(pool_vectors))]

        v = x_i + F * (x_pbest - x_i) + F * (x_r1 - x_r2)

        j_rand = np.random.randint(len(GENES))
        trial = np.empty_like(x_i)
        for j in range(len(GENES)):
            trial[j] = v[j] if ((np.random.rand() < CR) or (j == j_rand)) else x_i[j]
        trial = clamp_ind(trial)

        if ENABLE_REPAIR:
            trial = feasibility_repair(trial, x_i, model, scaler_X, scaler_Y)

        fit_trial, *_ = fitness_one(trial, model, scaler_X, scaler_Y, eps=eps)
        if fit_trial > fits[idx]:
            archive.append(pop[idx].copy())
            if len(archive) > ARCHIVE_MAX:
                del archive[np.random.randint(len(archive))]
            pop[idx] = trial
            fits[idx] = fit_trial


# ========= Diversity metrics =========
def diversity_metrics(pop):
    X = np.array(pop, dtype=np.float64)
    lo = np.array([VAR_BOUNDS[g][0] for g in GENES], dtype=np.float64)
    hi = np.array([VAR_BOUNDS[g][1] for g in GENES], dtype=np.float64)
    rng = np.maximum(hi - lo, 1e-12)
    Z = (X - lo) / rng
    s = np.sum(Z * Z, axis=1, keepdims=True)
    D2 = s + s.T - 2.0 * (Z @ Z.T)
    np.fill_diagonal(D2, 0.0)
    N = Z.shape[0]
    pair_mean = np.sqrt(np.sum(D2) / (N * (N - 1))) if N > 1 else 0.0
    stds = np.std(Z, axis=0, ddof=0)
    std_mean = float(np.mean(stds))
    return float(pair_mean), stds, std_mean


# ========= Visualization =========
def make_figures_from_history(outdir: str):
    df = pd.read_csv(os.path.join(outdir, "ga_history.csv"))
    ratio_col = "best_ratio_pair" if "best_ratio_pair" in df.columns else (
        "best_ratio" if "best_ratio" in df.columns else None
    )

    plt.figure(figsize=(6, 4))
    plt.plot(df["gen"], df["best_fit"], lw=2)
    plt.xlabel("Generation")
    plt.ylabel("Best Fy/G")
    plt.title("Convergence of Best Fy/G")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "A1_convergence_best_Fy_over_G.pdf"), dpi=300)

    if "feasible_%" in df.columns:
        plt.figure(figsize=(6, 4))
        plt.plot(df["gen"], df["feasible_%"], lw=2)
        plt.xlabel("Generation")
        plt.ylabel("Feasible population (%)")
        plt.title("Feasibility Rate")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "A2_feasibility_rate.png"), dpi=300)

    if "best_Fy_need(SF)" in df.columns:
        plt.figure(figsize=(6, 4))
        plt.plot(df["gen"], df["best_Fy"], label="Fy (best)", lw=2)
        plt.plot(df["gen"], df["best_Fy_need(SF)"], label="SF · Fy_min (best)", lw=2)
        plt.xlabel("Generation")
        plt.ylabel("Force (N)")
        plt.title("Safety Margin Tracking")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "A3_safety_margin_tracking.png"), dpi=300)

    if "best_alpha_deg" in df.columns:
        plt.figure(figsize=(6, 4))
        plt.plot(df["gen"], df["best_alpha_deg"], lw=2)
        plt.xlabel("Generation")
        plt.ylabel("Alpha (deg)")
        plt.title("Alpha Evolution")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "A4_alpha_evolution.png"), dpi=300)

    plt.figure(figsize=(7, 4.6))
    plt.plot(df["gen"], df["best_Fy"], label="Fy (best)")
    plt.plot(df["gen"], df["best_G"], label="G (best)")
    if ratio_col is not None:
        plt.plot(df["gen"], df[ratio_col], label="Fy/G (best)")
    plt.xlabel("Generation")
    plt.ylabel("Value")
    plt.title("Best Fy, G, and Fy/G")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "A5_best_Fy_G_ratio_over_gens.png"), dpi=300)

    stats_fp = os.path.join(outdir, "ga_violation_stats.csv")
    samp_fp = os.path.join(outdir, "ga_violation_samples.csv")
    if os.path.exists(stats_fp) and os.path.exists(samp_fp):
        dfs = pd.read_csv(stats_fp)
        dfv = pd.read_csv(samp_fp)

        plt.figure(figsize=(7, 4.8))
        plt.plot(dfs["gen"], dfs["viol_pos_mean"], label="mean (>0)", lw=2)
        plt.plot(dfs["gen"], dfs["viol_pos_p50"], label="p50 (>0)", lw=2)
        plt.plot(dfs["gen"], dfs["viol_pos_p90"], label="p90 (>0)", lw=2)
        plt.plot(dfs["gen"], dfs["viol_pos_max"], label="max (>0)", lw=2)
        plt.xlabel("Generation")
        plt.ylabel("Violation δ (N)")
        plt.title("Violation (δ = SF·Fy_min − Fy) over Generations")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "B2_violation_stats_curves.png"), dpi=300)

        last_gen = dfv["gen"].max()
        N_tail = max(5, int(dfs["gen"].max() * 0.2))
        tail = dfv[dfv["gen"] >= last_gen - N_tail + 1]["violation_delta"].values
        plt.figure(figsize=(6, 4.5))
        bins = 30
        if tail.size > 0:
            plt.hist(tail, bins=bins, density=True, alpha=0.6, label=f"last {N_tail} gens")
            if tail.size > 1:
                xs = np.linspace(min(0, tail.min()), tail.max() * 1.05 + 1e-6, 400)
                bw = (
                    1.06 * np.std(tail) * (tail.size ** (-1 / 5))
                    if np.std(tail) > 0
                    else 1.0
                )
                kde = np.zeros_like(xs)
                for v in tail:
                    kde += np.exp(-0.5 * ((xs - v) / bw) ** 2)
                kde /= (tail.size * bw * np.sqrt(2 * np.pi))
                plt.plot(xs, kde, lw=2)
        plt.axvline(0, ls="--", c="k", lw=1)
        plt.xlabel("Violation δ (N)")
        plt.ylabel("Density")
        plt.title("Violation Distribution (last generations)")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "B3_violation_hist_kde_tail.png"), dpi=300)

        S = max(1, int(len(dfs) // 12))
        gens = sorted(dfs["gen"].unique())
        sel_gens = gens[::S] if gens[-1] not in gens[::S] else gens[::S]
        data = []
        labels = []
        for g in sel_gens:
            v = dfv[dfv["gen"] == g]["violation_delta"].values
            v = v[v > 0]
            if v.size == 0:
                continue
            data.append(v)
            labels.append(str(g))
        if len(data) > 0:
            plt.figure(figsize=(max(6, 0.5 * len(labels) + 2), 4.5))
            plt.boxplot(data, labels=labels, showfliers=False)
            plt.xlabel("Generation")
            plt.ylabel("Positive Violation δ>0 (N)")
            plt.title("Positive Violation Boxplots over Generations")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(outdir, "B4_violation_boxplots_over_gens.png"), dpi=300)

        if len(dfs) > 0:
            arr = np.vstack(
                [dfs["viol_pos_p50"].to_numpy(), dfs["viol_pos_p90"].to_numpy()]
            )
            plt.figure(figsize=(8, 2.8))
            plt.imshow(arr, aspect="auto", cmap="viridis")
            xticks = range(0, len(dfs), S)
            plt.yticks([0, 1], ["p50", "p90"])
            plt.xticks(ticks=xticks, labels=dfs["gen"].iloc[::S])
            plt.colorbar(label="Violation δ (N)")
            plt.title("Violation Percentiles over Generations")
            plt.tight_layout()
            plt.savefig(os.path.join(outdir, "B5_violation_percentiles_heatmap.png"), dpi=300)

    div_fp = os.path.join(outdir, "ga_diversity.csv")
    if os.path.exists(div_fp):
        dv = pd.read_csv(div_fp)
        plt.figure(figsize=(6, 4))
        plt.plot(dv["gen"], dv["pairwise_L2_mean_norm"], lw=2)
        plt.xlabel("Generation")
        plt.ylabel("Mean pairwise L2 (normalized)")
        plt.title("Population Diversity: Pairwise Distance")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "D1_diversity_pairwise_L2.png"), dpi=300)

        plt.figure(figsize=(7, 4.6))
        for col, label in zip(
            ["std_norm_d", "std_norm_w", "std_norm_n1", "std_norm_R"],
            ["d (std/range)", "w (std/range)", "n1 (std/range)", "R (std/range)"],
        ):
            if col in dv.columns:
                plt.plot(dv["gen"], dv[col], label=label, lw=2)
        if "std_norm_mean" in dv.columns:
            plt.plot(
                dv["gen"],
                dv["std_norm_mean"],
                label="mean of dims",
                lw=2,
                linestyle="--",
            )
        plt.xlabel("Generation")
        plt.ylabel("Normalized STD")
        plt.title("Population Diversity: Per-Gene Normalized STD")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "D2_diversity_per_gene_std.png"), dpi=300)


# ========= Main procedure =========
def print_ind(ind):
    return f"d={ind[0]:.4f}, w={ind[1]:.4f}, n1={ind[2]:.4f}, R={ind[3]:.4f}"


def main():
    print(f"Output directory: {OUTDIR}")
    print("Hybrid optimization: Deb rules, epsilon-constraint annealing, local repair, diversity/violation logging")
    print("DE variant: current-to-pbest/1/bin with external archive")
    scaler_X, scaler_Y = load_scalers(save_dir)
    model = load_model(save_dir, weights_name, activation)

    with open(HIST_CSV, 'w', newline='', encoding='utf-8-sig') as f:
        csv.writer(f).writerow(
            [
                'gen',
                'best_fit',
                'best_Fy',
                'best_G',
                'best_ratio_pair',
                'best_Fy_need(SF)',
                'best_alpha_deg',
                'd',
                'w',
                'n1',
                'R',
                'feasible_%',
                'mean_fit_ok',
            ]
        )
    with open(VIO_STATS_CSV, "w", newline="", encoding="utf-8-sig") as f:
        csv.writer(f).writerow(
            [
                "gen",
                "feasible_pct",
                "n_pop",
                "viol_pos_mean",
                "viol_pos_p50",
                "viol_pos_p90",
                "viol_pos_max",
            ]
        )
    with open(VIO_SAMPLES_CSV, "w", newline="", encoding="utf-8-sig") as f:
        csv.writer(f).writerow(["gen", "violation_delta"])
    with open(DIV_CSV, "w", newline="", encoding="utf-8-sig") as f:
        csv.writer(f).writerow(
            [
                "gen",
                "pairwise_L2_mean_norm",
                "std_norm_d",
                "std_norm_w",
                "std_norm_n1",
                "std_norm_R",
                "std_norm_mean",
            ]
        )

    pop = init_population(model, scaler_X, scaler_Y, POP_SIZE)
    archive = []

    best_fit = -1e-18
    best_ind = None
    best_vals = (np.nan, np.nan, np.nan, np.nan, np.nan)

    t0 = time.time()
    for gen in range(1, N_GENERATIONS + 1):
        eps_t = epsilon_at_gen(gen)

        fits, Fy_all, G_all, Fy_need_all, alpha_all = population_fitness(
            pop, model, scaler_X, scaler_Y, eps=eps_t
        )

        de_step_inplace_pbest(
            pop,
            fits,
            model,
            scaler_X,
            scaler_Y,
            archive,
            portion=DE_PORTION,
            f_range=DE_F_RANGE,
            cr_range=DE_CR_RANGE,
            pbest_frac=DE_PBEST_FRAC,
            eps=eps_t,
        )

        fits, Fy_all, G_all, Fy_need_all, alpha_all = population_fitness(
            pop, model, scaler_X, scaler_Y, eps=eps_t
        )

        ok_mask = fits > -1e-11
        idx_best = int(np.argmax(fits))
        gen_best_fit = float(fits[idx_best])
        gen_best_ind = pop[idx_best].copy()
        ratio_pair = float(Fy_all[idx_best] / max(G_all[idx_best], EPS)) if ok_mask[idx_best] else float('nan')
        gen_best_vals = (
            float(Fy_all[idx_best]),
            float(G_all[idx_best]),
            ratio_pair,
            float(SAFETY_FACTOR * Fy_need_all[idx_best]),
            float(alpha_all[idx_best]),
        )

        delta = SAFETY_FACTOR * Fy_need_all - Fy_all
        viol_pos = delta[delta > 0.0]
        feasible_pct = float(np.mean(delta <= 0.0) * 100.0)
        with open(VIO_STATS_CSV, "a", newline="", encoding="utf-8-sig") as f:
            row = [
                gen,
                feasible_pct,
                len(delta),
                float(np.mean(viol_pos)) if viol_pos.size > 0 else 0.0,
                float(np.percentile(viol_pos, 50)) if viol_pos.size > 0 else 0.0,
                float(np.percentile(viol_pos, 90)) if viol_pos.size > 0 else 0.0,
                float(np.max(viol_pos)) if viol_pos.size > 0 else 0.0,
            ]
            csv.writer(f).writerow(row)
        SNAP_N = min(400, len(delta))
        idxs = np.random.choice(len(delta), size=SNAP_N, replace=False)
        with open(VIO_SAMPLES_CSV, "a", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            for j in idxs:
                w.writerow([gen, float(delta[j])])

        pair_mean, stds, std_mean = diversity_metrics(pop)
        with open(DIV_CSV, "a", newline="", encoding="utf-8-sig") as f:
            row = [
                gen,
                pair_mean,
                stds[GENES.index('d')],
                stds[GENES.index('w')],
                stds[GENES.index('n1')],
                stds[GENES.index('R')],
                std_mean,
            ]
            csv.writer(f).writerow(row)

        if gen_best_fit > best_fit:
            best_fit = gen_best_fit
            best_ind = gen_best_ind.copy()
            best_vals = gen_best_vals

        feasible_ratio = float(np.mean(ok_mask) * 100.0)
        mean_fit = float(np.mean(fits[ok_mask])) if np.any(ok_mask) else float('nan')

        if gen % PRINT_EVERY == 0:
            Fy_b, G_b, r_b, Fy_need_sf_b, a_b = gen_best_vals
            print(
                f"[Gen {gen:03d}] Feasible={feasible_ratio:5.1f}% | Best(Fy/G)={gen_best_fit:10.6f} | "
                f"{print_ind(gen_best_ind)} | Fy={Fy_b:.4f}N, G={G_b:.4f}N, "
                f"Fy_need(SF)={Fy_need_sf_b:.4f}N, alpha={a_b * 180 / math.pi:.3f} deg | "
                f"MeanFit={mean_fit:.6f} | eps={eps_t:.3f}N | div_pair={pair_mean:.3f}, std_mean={std_mean:.3f} | "
                f"archive_size={len(archive)}"
            )

        with open(HIST_CSV, 'a', newline='', encoding='utf-8-sig') as f:
            csv.writer(f).writerow(
                [
                    gen,
                    gen_best_fit,
                    gen_best_vals[0],
                    gen_best_vals[1],
                    gen_best_vals[2],
                    gen_best_vals[3],
                    gen_best_vals[4] * 180 / math.pi,
                    gen_best_ind[0],
                    gen_best_ind[1],
                    gen_best_ind[2],
                    gen_best_ind[3],
                    feasible_ratio,
                    mean_fit,
                ]
            )

        elite_idx = np.argsort(fits)[-ELITE_KEEP:]
        elites = [pop[i].copy() for i in elite_idx]
        new_pop = []
        while len(new_pop) < POP_SIZE - ELITE_KEEP:
            p1 = tournament_select(pop, fits, TOURN_K, Fy_all, Fy_need_all, eps=eps_t)
            p2 = tournament_select(pop, fits, TOURN_K, Fy_all, Fy_need_all, eps=eps_t)
            c1, c2 = uniform_crossover(p1, p2, CROSS_P)
            c1 = mutate(c1, MUT_P)
            if ENABLE_REPAIR:
                c1 = feasibility_repair(c1, p1, model, scaler_X, scaler_Y)
            new_pop.append(c1)
            if len(new_pop) < POP_SIZE - ELITE_KEEP:
                c2 = mutate(c2, MUT_P)
                if ENABLE_REPAIR:
                    c2 = feasibility_repair(c2, p2, model, scaler_X, scaler_Y)
                new_pop.append(c2)
        pop = new_pop + elites

    t1 = time.time()
    print("\n========== Final result ==========")
    if best_ind is None or best_fit <= -1e-11:
        print("No feasible solution that satisfies the constraints was found.")
    else:
        Fy_b, G_b, r_b, Fy_need_sf_b, a_b = best_vals
        ratio_pair = Fy_b / max(G_b, EPS)
        print(f"Best individual: {print_ind(best_ind)}")
        print(f"Prediction: Fy={Fy_b:.6f} N | G={G_b:.6f} N | Fy/G={ratio_pair:.6f}")
        print(
            f"Constraint: Fy_need(SF={SAFETY_FACTOR})={Fy_need_sf_b:.6f} N | "
            f"alpha={a_b * 180 / math.pi:.6f} deg | satisfied={Fy_b >= Fy_need_sf_b}"
        )
        print(f"Total runtime: {t1 - t0:.2f} s")
    print(f"History CSV: {os.path.abspath(HIST_CSV)}")

    print("Generating figures...")
    make_figures_from_history(OUTDIR)
    print(f"Figures saved to: {os.path.abspath(OUTDIR)}")


if __name__ == '__main__':
    main()
