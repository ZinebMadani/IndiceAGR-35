"""
06_Convex_Optimization.py
--------------------------------------------------------------
Construction de l'indice par optimisation convexe bi-penalisee
(section II.3, "Optimisation : modelisation retenue (modele
unique)", equation eq:erc_bipenalise) :

    w*(lambda_sys, lambda_fx) =
        argmin_w  w' Sigma_clean w
                  + lambda_sys * s'w
                  + lambda_fx  * s^{fx}'w
        s.c.      1'w = 1,  0 <= w <= w_max

ou :
    - Sigma_clean est la matrice de covariance nettoyee par RMT
      (module 04_RMT_Cleaning.py) ;
    - s est le score de risque systemique normalise, issu du
      Delta-CoVaR (module 05_Systemic_Risk.py) ;
    - s^fx est le score d'exposition au risque de change normalise,
      issu du beta sur le facteur de change commun DCC-GARCH/RMT+PCA
      (module 02_FX_Adjusted_Index.py).

Conformement a la section "Optimisation : modelisation retenue
(modele unique)" du memoire, ce module retient UNE SEULE
modelisation integrant conjointement les deux dimensions de risque :
il n'y a plus de penalisation systemique seule (ancien
erc_penalized_weights) traitee separement du risque de change — les
deux sont desormais deux termes lineaires du meme programme convexe.

A (lambda_sys, lambda_fx) = (0, 0), le programme se reduit au
portefeuille a variance minimale (Proposition erc-bipenalise :
existence et unicite de la solution pour tout Sigma_clean SDP et
tout (lambda_sys, lambda_fx) >= (0, 0), demontree par convexite
stricte de l'objectif sur le polytope compact des poids).

Portefeuilles de reference compares au portefeuille bi-penalise
(section "Portefeuilles de reference") :
    - Market Cap (poids fournis en entree, non contraints par w_max) ;
    - Min-Variance = bi-penalise a (0, 0) ;
    - ERC (Equal Risk Contribution), obtenu par le programme
      log-barriere de Spinu (2013), resolu par la methode de Newton
      (initialisation equiponderee, tolerance 1e-9, 200 iterations
      maximum), SANS contrainte de plafond w_max.

RENOMME depuis 05_Convex_Optimization.py : le numero de module 05
est maintenant pris par 05_Systemic_Risk.py.

Convention de numerotation des graphiques : prefixe "48_" et
suivants (module construction : 01-27b, module FX : 27-31b, module
DCC-GARCH : 32-35, module RMT : 36-40, module risque systemique :
41-47, module optimisation : 48+).

Ce module ne touche ni au classeur Excel ni au Dashboard :
01_Construction_Indice.py pilote l'export complet.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PRIMARY_COLOR = "#064E3B"

plt.rcParams["axes.prop_cycle"] = plt.cycler(
    color=[PRIMARY_COLOR]
)

import cvxpy as cp

from utils import save_chart

# ================================================================
# PALETTE DE COULEURS DES GRAPHIQUES
# ================================================================

COLORS = {
    "Market_Cap": "#1F77B4",
    "Min_Variance": "#FF7F0E",        # Orange
    "ERC": "#2CA02C",                 # Vert
    "ERC_Bipenalized": "#D62728",     # Rouge
    "Reference": "#555555",           # Gris foncé
}


# ================================================================
# PROGRAMME BI-PENALISE (eq:erc_bipenalise)
# ================================================================

def bipenalized_weights(
    Sigma: np.ndarray,
    s_sys: np.ndarray,
    s_fx: np.ndarray,
    lam_sys: float = 0.0,
    lam_fx: float = 0.0,
    w_max: float = 0.15,
):
    """
    Resout le programme d'optimisation convexe bi-penalise
    (eq:erc_bipenalise) :

        min_w  w' Sigma w + lam_sys * s_sys'w + lam_fx * s_fx'w

        s.c.   somme_i w_i = 1
               0 <= w_i <= w_max

    A lam_sys = lam_fx = 0, ce programme est strictement le
    portefeuille a variance minimale : min_variance_weights n'est
    donc plus une fonction separee, c'est ce meme solveur applique
    a (0, 0).

    Les deux termes de penalite sont mis a l'echelle du terme de
    risque quadratique (proxy : variance du portefeuille equipondere),
    afin qu'une valeur de lambda de l'ordre de 1 corresponde a une
    penalisation raisonnable pour les deux dimensions de risque.
    """

    N = Sigma.shape[0]

    s_sys = np.asarray(s_sys, dtype=float)
    s_fx = np.asarray(s_fx, dtype=float)

    if len(s_sys) != N or len(s_fx) != N:
        raise ValueError(
            "s_sys et s_fx doivent avoir la même dimension que Sigma."
        )

    if N * w_max < 1:
        raise ValueError(
            f"Contrainte irréalisable : {N} banques avec "
            f"w_max={w_max:.2%}. Il faut N*w_max >= 1."
        )

    # --------------------------------------------------------------
    # Echelle des termes de pénalité (proxy : portefeuille équipondéré)
    # --------------------------------------------------------------

    w_eq = np.ones(N) / N
    risk_scale = float(w_eq @ Sigma @ w_eq)

    if risk_scale <= 0:
        risk_scale = 1e-8  # garde-fou numérique

    w = cp.Variable(N)

    risk = cp.quad_form(w, cp.psd_wrap(Sigma))

    penalty_sys = lam_sys * risk_scale * (s_sys @ w)
    penalty_fx = lam_fx * risk_scale * (s_fx @ w)

    objective = cp.Minimize(risk + penalty_sys + penalty_fx)

    constraints = [
        cp.sum(w) == 1,
        w >= 0,
        w <= w_max,
    ]

    prob = cp.Problem(objective, constraints)

    prob.solve(
        solver=cp.OSQP,
        eps_abs=1e-10,
        eps_rel=1e-10,
        max_iter=100000,
    )

    if w.value is None:
        raise RuntimeError(
            "L'optimisation bi-pénalisée n'a pas trouvé de solution "
            f"(lam_sys={lam_sys}, lam_fx={lam_fx})."
        )

    return np.asarray(w.value).flatten()


# ================================================================
# CONTRIBUTIONS AU RISQUE
# ================================================================

def risk_contributions(
    Sigma: np.ndarray,
    w: np.ndarray
) -> np.ndarray:
    """
    Contribution au risque de la banque i :

        RC_i = w_i * (Sigma w)_i / sigma(w)

    avec sigma(w) = sqrt(w' Sigma w). La somme des contributions
    est égale à la volatilité du portefeuille.
    """

    variance = w @ Sigma @ w

    if variance <= 0:
        return np.zeros_like(w)

    sigma_w = np.sqrt(variance)

    return (w * (Sigma @ w)) / sigma_w


# ================================================================
# ERC PAR PROGRAMME LOG-BARRIERE (SPINU, 2013), METHODE DE NEWTON
# ================================================================

def erc_log_barrier_weights(
    Sigma: np.ndarray,
    c: float = 1.0,
    tol: float = 1e-9,
    max_iter: int = 200,
):
    """
    Portefeuille Equal Risk Contribution (ERC), SANS contrainte de
    plafond w_max, obtenu par la reformulation convexe de Spinu (2013) :

        x* = argmin_{x>0}  (1/2) x' Sigma x - c * sum_i ln(x_i)

    resolue par la methode de Newton (initialisation equiponderee,
    tolerance 1e-9, 200 iterations maximum), puis renormalisee :

        w = x* / sum(x*)

    La condition du premier ordre du programme log-barriere,
    (Sigma x)_i = c / x_i pour tout i, equivaut a x_i (Sigma x)_i = c
    pour tout i : les contributions au risque brutes sont donc
    egalisees a c, quel que soit c > 0 (c n'affecte pas w apres
    renormalisation, seule l'echelle de x* en depend).

    Un pas de Newton amorti (backtracking) garantit x > 0 a chaque
    iteration, condition necessaire au domaine du log-barriere.
    """

    N = Sigma.shape[0]

    # Symétrisation numérique de Sigma
    Sigma = (
        np.asarray(Sigma, dtype=float)
        + np.asarray(Sigma, dtype=float).T
    ) / 2.0

    x = np.ones(N) / N  # initialisation équipondérée

    for _ in range(max_iter):

        grad = Sigma @ x - c / x
        hess = Sigma + np.diag(c / x ** 2)

        try:
            step = np.linalg.solve(hess, grad)
        except np.linalg.LinAlgError:
            hess_reg = hess + 1e-10 * np.eye(N)
            step = np.linalg.solve(hess_reg, grad)

        # Backtracking : garantir x_new > 0 (domaine du log-barrière)
        alpha = 1.0
        x_new = x - alpha * step

        while np.any(x_new <= 0) and alpha > 1e-12:
            alpha *= 0.5
            x_new = x - alpha * step

        if np.any(x_new <= 0):
            # Echec du backtracking : on garde la derniere iteration valide
            break

        if np.linalg.norm(x_new - x) < tol:
            x = x_new
            break

        x = x_new

    w = x / x.sum()

    return w


# ================================================================
# FRONTIERE BI-PENALISEE
# ================================================================

def bipenalized_frontier(
    Sigma: np.ndarray,
    s_sys: np.ndarray,
    s_fx: np.ndarray,
    lambda_pairs,
    w_max: float = 0.15,
):
    """
    Resout le programme bi-penalise pour une grille de couples
    (lambda_sys, lambda_fx), afin d'isoler la contribution de chaque
    dimension de risque a l'arbitrage global (section "Choix des
    lambdas et frontiere risque / exposition") :

        - lambda_fx = 0 : penalisation systemique seule
        - lambda_sys = 0 : penalisation change seule
        - lambda_sys > 0 et lambda_fx > 0 : combinaison des deux

    Retourne un DataFrame avec, pour chaque couple : Lambda_Sys,
    Lambda_FX, Variance, Volatility_Annualized, Systemic_Exposure
    (s'w) et FX_Exposure (s^fx'w).
    """

    rows = []

    for lam_sys, lam_fx in lambda_pairs:

        w = bipenalized_weights(
            Sigma, s_sys, s_fx,
            lam_sys=lam_sys,
            lam_fx=lam_fx,
            w_max=w_max,
        )

        var_w = float(w @ Sigma @ w)

        rows.append({
            "Lambda_Sys": lam_sys,
            "Lambda_FX": lam_fx,
            "Variance": var_w,
            "Volatility_Annualized": np.sqrt(var_w) * np.sqrt(252),
            "Systemic_Exposure": float(s_sys @ w),
            "FX_Exposure": float(s_fx @ w),
        })

    return pd.DataFrame(rows)


# ================================================================
# INDICATEURS DE PORTEFEUILLE
# ================================================================

def portfolio_indicators(
    w: np.ndarray,
    Sigma: np.ndarray,
    s_sys: np.ndarray = None,
    s_fx: np.ndarray = None,
) -> dict:
    """
    Indicateurs comparatifs (section "Indicateurs comparatifs des
    portefeuilles") : variance/volatilite, HHI(w) = sum_i w_i^2,
    nombre effectif de banques N_eff(w) = 1/HHI(w), Top5/Top10, et,
    si fournis, l'exposition systemique s'w et l'exposition change
    s^fx'w.
    """

    variance = float(w @ Sigma @ w)
    volatility = float(np.sqrt(max(variance, 0)))
    annual_volatility = volatility * np.sqrt(252)

    hhi = float(np.sum(w ** 2))
    effective_n = 1 / hhi if hhi > 0 else np.nan

    top5 = float(np.sum(np.sort(w)[::-1][:5]))
    top10 = float(np.sum(np.sort(w)[::-1][:10]))

    indicators = {
        "Variance_Quotidienne": variance,
        "Volatilite_Quotidienne": volatility,
        "Volatilite_Annualisee": annual_volatility,
        "HHI": hhi,
        "Nombre_Effectif_Banques": effective_n,
        "Top5_Poids": top5,
        "Top10_Poids": top10,
    }

    if s_sys is not None:
        indicators["Exposition_Systemique"] = float(np.asarray(s_sys) @ w)

    if s_fx is not None:
        indicators["Exposition_Change"] = float(np.asarray(s_fx) @ w)

    return indicators


# ================================================================
# PIPELINE COMPLET D'OPTIMISATION CONVEXE BI-PENALISEE
# ================================================================

def run_convex_optimization(
    returns_df,
    Sigma_clean,
    systemic_score,
    fx_exposure_score,
    info=None,
    w_max=0.15,
    lam_sys=1.0,
    lam_fx=1.0,
    lambda_pairs=None,
    other_indices=None,
    market_cap_weights=None, 
):
    """
    Exécute l'ensemble du pipeline d'optimisation bi-pénalisée :

    1. Portefeuille Min-Variance (= bi-pénalisé à (0, 0))
    2. Portefeuille ERC (log-barrière, sans plafond w_max)
    3. Portefeuille bi-pénalisé retenu (lam_sys, lam_fx)
    4. Frontière bi-pénalisée (systémique seule / change seule / combinée)
    5. Comparaison des pondérations
    6. Statistiques comparatives (variance, HHI, expositions)
    7. Construction des indices Base 100
    8. Graphiques

    Parameters
    ----------
    returns_df : pandas.DataFrame
        Rendements logarithmiques des banques.

    Sigma_clean : pandas.DataFrame
        Matrice de covariance nettoyée par RMT (sortie de
        04_RMT_Cleaning.py, run_rmt_cleaning()["Sigma_clean"]).

    weights : pandas.Series
        Pondérations Market Cap utilisées comme référence.

    systemic_score : pandas.Series
        Score systémique normalisé s (sortie de
        04_Systemic_Risk.py, run_systemic_risk()["systemic_score"]),
        indexé comme returns_df.columns.

    fx_exposure_score : pandas.Series
        Score d'exposition au change normalisé s^fx (sortie de
        02_FX_Adjusted_Index.py, run_fx_adjusted_index()["fx_exposure_score"]),
        indexé comme returns_df.columns.

    info : pandas.DataFrame or None
        Informations sur les banques (colonne Country notamment).

    w_max : float
        Poids maximal autorisé par banque (portefeuilles contraints
        Min-Variance et bi-pénalisé ; l'ERC log-barrière n'est PAS
        soumis à ce plafond, conformément au mémoire).

    lam_sys, lam_fx : float
        Couple (lambda_sys, lambda_fx) du portefeuille bi-pénalisé
        retenu comme résultat principal.

    lambda_pairs : list of (float, float) or None
        Grille de couples (lambda_sys, lambda_fx) pour la frontière.
        Par défaut : pénalisation systémique seule, pénalisation
        change seule, et une combinaison des deux (section "Choix
        des lambdas et frontière risque / exposition").

    other_indices : dict or None
        Indices existants à comparer (ex. Market_Cap, FX_Adjusted...).

    market_cap_weights : pandas.Series or None
        Poids Market Cap par ticker (ex. weights_info.set_index("Ticker")["Weight"]).
        Utilisés pour la barre « Pondération par capi » du graphique HHI.
    """

    print("\n" + "=" * 70)
    print("OPTIMISATION CONVEXE BI-PENALISEE DE L'INDICE")
    print("=" * 70)

    tickers = list(Sigma_clean.columns)
    N = len(tickers)

    if w_max <= 0 or w_max > 1:
        raise ValueError("w_max doit être compris entre 0 et 1.")

    if N * w_max < 1:
        raise ValueError(
            f"Le plafond w_max={w_max:.2%} est trop faible pour "
            f"{N} banques. Il faut au minimum w_max >= {1/N:.2%}."
        )

    print(f"\nNombre de banques : {N}")
    print(f"Poids maximal autorisé (Min-Variance / bi-pénalisé) : {w_max:.2%}")
    print(f"Couple retenu : (lambda_sys, lambda_fx) = ({lam_sys:g}, {lam_fx:g})")

    # ------------------------------------------------------------
    # Alignement des scores sur les tickers de Sigma_clean
    # ------------------------------------------------------------

    s_sys = systemic_score.reindex(tickers)

    if fx_exposure_score is None:
        print(
            "Attention : score FX indisponible. "
            "Penalité FX désactivée."
        )

        s_fx = pd.Series(
            0.0,
            index=tickers
        )

    else:
        s_fx = fx_exposure_score.reindex(tickers).fillna(0)

    if fx_exposure_score is None:

        print(
            "Attention : score FX absent. "
            "Utilisation d'un vecteur neutre."
        )

        s_fx = pd.Series(
            0,
            index=tickers
        )

    else:

        s_fx = (
            fx_exposure_score
            .reindex(tickers)
            .fillna(
                fx_exposure_score.mean()
            )
        )

    for name, s in (("systemic_score", s_sys), ("fx_exposure_score", s_fx)):

        if s.isna().any():

            missing = list(s[s.isna()].index)

            raise ValueError(
                f"{name} manquant pour : " + ", ".join(missing)
            )

    Sigma = Sigma_clean.values

    # Régularisation numérique
    Sigma_reg = (
        Sigma
        + 1e-10 * np.trace(Sigma) / N * np.eye(N)
    )

    # ============================================================
    # 1. PORTEFEUILLE A VARIANCE MINIMALE (bi-pénalisé à (0, 0))
    # ============================================================

    print("\nCalcul du portefeuille à variance minimale (lam_sys=lam_fx=0)...")

    w_mv = bipenalized_weights(
        Sigma_reg, s_sys.values, s_fx.values,
        lam_sys=0.0, lam_fx=0.0, w_max=w_max,
    )

    w_mv_series = pd.Series(w_mv, index=tickers, name="Min_Variance")

    print(f"Poids maximum obtenu : {w_mv.max():.4%}")
    print(f"Volatilité quotidienne (Min-Variance) : {np.sqrt(w_mv @ Sigma @ w_mv):.6f}")

    # ============================================================
    # 2. PORTEFEUILLE ERC (LOG-BARRIERE, SANS PLAFOND w_max)
    # ============================================================

    print("\nCalcul du portefeuille Equal Risk Contribution (ERC, log-barrière)...")

    w_erc = erc_log_barrier_weights(Sigma_reg)

    w_erc_series = pd.Series(w_erc, index=tickers, name="ERC")

    rc_erc = risk_contributions(Sigma, w_erc)

    print(f"Poids maximum obtenu : {w_erc.max():.4%}")
    print(f"Écart-type des contributions au risque (ERC) : {rc_erc.std():.8f}")
    print("Plus cet écart est proche de zéro, plus les contributions au risque sont égalisées.")

    # ============================================================
    # 3. PORTEFEUILLE BI-PENALISE RETENU
    # ============================================================

    print(
        "\nCalcul du portefeuille bi-pénalisé "
        f"(lam_sys={lam_sys:g}, lam_fx={lam_fx:g})..."
    )

    w_bipen = bipenalized_weights(
        Sigma_reg, s_sys.values, s_fx.values,
        lam_sys=lam_sys, lam_fx=lam_fx, w_max=w_max,
    )

    w_bipen_series = pd.Series(
        w_bipen, index=tickers,
        name=f"ERC_Bipenalized_sys{lam_sys:g}_fx{lam_fx:g}",
    )

    print(f"Poids maximum obtenu : {w_bipen.max():.4%}")
    print(f"Exposition systémique (s'w) : {s_sys.values @ w_bipen:.6f}")
    print(f"Exposition change (s^fx'w)  : {s_fx.values @ w_bipen:.6f}")

    # ============================================================
    # 4. FRONTIERE BI-PENALISEE
    # ============================================================

    if lambda_pairs is None:

        sys_only = [(lam, 0.0) for lam in np.linspace(0, 5, 11)]
        fx_only = [(0.0, lam) for lam in np.linspace(0, 5, 11)]
        combined = [(lam, lam) for lam in np.linspace(0, 5, 11)]

        lambda_pairs = sys_only + fx_only + combined

    print(
        f"\nCalcul de la frontière bi-pénalisée ({len(lambda_pairs)} couples)..."
    )

    frontier_df = bipenalized_frontier(
        Sigma_reg, s_sys.values, s_fx.values,
        lambda_pairs, w_max=w_max,
    )

    # ============================================================
    # 5. TABLEAU DE COMPARAISON DES PONDERATIONS
    # ============================================================

    print("\nConstruction du tableau de comparaison des pondérations...")

    weights_comparison = pd.concat(
        [w_mv_series, w_erc_series, w_bipen_series],
        axis=1,
    )

    weights_comparison.index.name = "Ticker"

    if info is not None and "Country" in info.columns:

        country_map = info.set_index("Ticker")["Country"]

        weights_comparison.insert(
            0, "Country", weights_comparison.index.map(country_map)
        )

    weights_comparison = weights_comparison.reset_index()

    # ============================================================
    # 6. STATISTIQUES COMPARATIVES
    # ============================================================

    print("\nCalcul des statistiques comparatives...")

    if market_cap_weights is not None:
        w_mc_series = (
            pd.Series(market_cap_weights).reindex(tickers).fillna(0.0)
        )
        w_mc_series = w_mc_series / w_mc_series.sum()  # renormalise si sous-univers
        w_mc = w_mc_series.values
    else:
        w_mc = None
        print("Attention : market_cap_weights non fourni, "
              "Market_Cap absent des statistiques.")

    portfolio_names = ["Min_Variance", "ERC", w_bipen_series.name]
    portfolio_vectors = [w_mv, w_erc, w_bipen]

    if w_mc is not None:
        portfolio_names.insert(0, "Market_Cap")
        portfolio_vectors.insert(0, w_mc)

    stats_rows = []

    for name, w_vec in zip(portfolio_names, portfolio_vectors):

        row = {"Portfolio": name}

        row.update(
            portfolio_indicators(w_vec, Sigma, s_sys=s_sys.values, s_fx=s_fx.values)
        )

        stats_rows.append(row)

    portfolio_stats = pd.DataFrame(stats_rows)

    # ============================================================
    # 7. CONSTRUCTION DES INDICES BASE 100
    # ============================================================

    print("\nConstruction des indices Base 100...")

    returns_universe = returns_df[tickers].copy()

    def build_index(w_vec, name):
        """
        r_index,t = somme_i w_i r_i,t
        I_t = 100 * exp(somme_{s<=t} r_index,s)
        """

        w_series = pd.Series(w_vec, index=tickers)

        idx_returns = returns_universe.mul(w_series, axis=1).sum(axis=1)

        idx_level = 100 * np.exp(idx_returns.cumsum())
        idx_level.name = name

        return idx_level

    index_min_variance = build_index(w_mv, "Index_Min_Variance")
    index_erc = build_index(w_erc, "Index_ERC")
    index_bipenalized = build_index(w_bipen, "Index_ERC_Bipenalized")

    # ============================================================
    # 8. COMPARAISON DES INDICES
    # ============================================================

    all_indices = {
        "Min_Variance": index_min_variance,
        "ERC": index_erc,
        "ERC_Bipenalized": index_bipenalized,
    }

    if other_indices is not None:

        for name, series in other_indices.items():

            if series is None:
                continue

            if isinstance(series, pd.DataFrame):

                if series.shape[1] != 1:
                    continue

                series = series.iloc[:, 0]

            if not isinstance(series, pd.Series):
                continue

            all_indices[name] = series

    index_comparison = pd.concat(all_indices, axis=1).dropna(how="all")

    index_comparison_common = index_comparison.dropna(how="any")

    if len(index_comparison_common) > 0:

        first_common_date = index_comparison_common.index[0]

        rebased = index_comparison.copy()

        for column in rebased.columns:

            series = rebased[column].dropna()

            if len(series) == 0:
                continue

            base_value = (
                series.loc[first_common_date]
                if first_common_date in series.index
                else series.iloc[0]
            )

            rebased.loc[series.index, column] = series / base_value * 100

        index_comparison = rebased

    # ============================================================
    # 9. GRAPHIQUES
    # ============================================================

    print("\nCréation des graphiques d'optimisation convexe...")

    # ------------------------------------------------------------
    # 9.1 COMPARAISON DES PONDERATIONS (Min-Var / ERC / Bi-pénalisé)
    # ------------------------------------------------------------

    bipen_col = w_bipen_series.name   # ex. "ERC_Bipenalized_sys1_fx1"

    cols_to_plot = ["Min_Variance", "ERC", bipen_col]

    # Libellés et couleurs (la clé de COLORS pour le bi-pénalisé est "ERC_Bipenalized")
    labels = {
        "Min_Variance": "Min-Variance",
        "ERC": "ERC",
        bipen_col: "Min-Variance bi-pénalisé",
    }
    colors = {
        "Min_Variance": COLORS["Min_Variance"],
        "ERC": COLORS["ERC"],
        bipen_col: COLORS["ERC_Bipenalized"],
    }

    plot_df = (
        weights_comparison
        .set_index("Ticker")[cols_to_plot]
        .sort_values("Min_Variance", ascending=False)
    )

    x = np.arange(len(plot_df))
    n_series = plot_df.shape[1]
    width = 0.8 / n_series

    plt.figure(figsize=(15, 6))

    for i, column in enumerate(plot_df.columns):
        plt.bar(
            x + i * width, plot_df[column], width=width,
            label=labels[column], color=colors[column],
        )

    plt.xticks(x + width * (n_series - 1) / 2, plot_df.index, rotation=90)
    plt.ylabel("Poids (échelle log)")
    plt.xlabel("Banque")
    plt.title("Comparaison des pondérations — Min-Variance / ERC / Min-Variance bi-pénalisé")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.yscale("log")

    save_chart("48_Weights_Comparison_MinVar_ERC_Bipenalized")
    plt.close()

    # ------------------------------------------------------------
    # 9.2 CONTRIBUTIONS AU RISQUE ERC
    # ------------------------------------------------------------

    rc_sorted = pd.Series(rc_erc, index=tickers).sort_values(ascending=False)

    plt.figure(figsize=(14, 6))

    plt.bar(rc_sorted.index, rc_sorted.values)

    plt.axhline(
        rc_erc.sum() / len(tickers), linestyle="--", linewidth=1.5,
        color="black", label="Contribution moyenne (cible ERC)",
    )

    plt.xticks(rotation=90)
    plt.ylabel("Contribution au risque")
    plt.xlabel("Banque")
    plt.title("Contributions au risque du portefeuille ERC")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)

    save_chart("49_ERC_Risk_Contributions")
    plt.close()

    # ------------------------------------------------------------
    # 9.3 FRONTIERES : SYSTEMIQUE SEULE / CHANGE SEULE
    # ------------------------------------------------------------

    frontier_sys_only = frontier_df[frontier_df["Lambda_FX"] == 0.0]
    frontier_fx_only = frontier_df[frontier_df["Lambda_Sys"] == 0.0]

    if len(frontier_sys_only) > 0:

        plt.figure(figsize=(10, 7))

        plt.plot(
            frontier_sys_only["Systemic_Exposure"],
            frontier_sys_only["Volatility_Annualized"],
            marker="o", linewidth=1.5,
        )

        for _, row in frontier_sys_only.iloc[::3].iterrows():

            plt.annotate(
                f"λ_sys={row['Lambda_Sys']:.2g}",
                (row["Systemic_Exposure"], row["Volatility_Annualized"]),
                fontsize=7, textcoords="offset points", xytext=(5, 5),
            )

        plt.xlabel("Exposition au risque systémique (s'w)")
        plt.ylabel("Volatilité annualisée")
        plt.title("Frontière risque / exposition systémique (pénalisation systémique seule)")
        plt.grid(True, alpha=0.3)

        save_chart("50_Risk_Frontier_Systemic_Only")
        plt.close()

    if len(frontier_fx_only) > 0:

        plt.figure(figsize=(10, 7))

        plt.plot(
            frontier_fx_only["FX_Exposure"],
            frontier_fx_only["Volatility_Annualized"],
            marker="o", linewidth=1.5, color="darkorange",
        )

        for _, row in frontier_fx_only.iloc[::3].iterrows():

            plt.annotate(
                f"λ_fx={row['Lambda_FX']:.2g}",
                (row["FX_Exposure"], row["Volatility_Annualized"]),
                fontsize=7, textcoords="offset points", xytext=(5, 5),
            )

        plt.xlabel("Exposition au risque de change (s^fx'w)")
        plt.ylabel("Volatilité annualisée")
        plt.title("Frontière risque / exposition change (pénalisation change seule)")
        plt.grid(True, alpha=0.3)

        save_chart("51_Risk_Frontier_FX_Only")
        plt.close()

    # ------------------------------------------------------------
    # 9.4 ERC VS ERC BI-PENALISE
    # ------------------------------------------------------------

    plot_df2 = pd.DataFrame({
        "ERC": w_erc_series,
        w_bipen_series.name: w_bipen_series,
    }).sort_values("ERC", ascending=False)

    x2 = np.arange(len(plot_df2))
    width2 = 0.4

    plt.figure(figsize=(15, 6))

    plt.bar(
        x2 - width2 / 2, plot_df2["ERC"], width=width2,
        label="ERC", color=COLORS["ERC"],
    )

    plt.bar(
        x2 + width2 / 2, plot_df2[w_bipen_series.name], width=width2,
        label="Min-Variance bi-pénalisé", color=COLORS["ERC_Bipenalized"],
    )

    plt.xticks(x2, plot_df2.index, rotation=90)
    plt.ylabel("Poids")
    plt.xlabel("Banque")
    plt.title(
        f"ERC vs Min-Variance bi-pénalisé (lambda_sys={lam_sys:g}, lambda_fx={lam_fx:g})"
    )
    plt.legend()
    plt.grid(axis="y", alpha=0.3)

    save_chart("52_Weights_ERC_vs_Bipenalized")
    plt.close()

    # ------------------------------------------------------------
    # 9.5 COMPARAISON DES INDICES
    # ------------------------------------------------------------

    fig_comp, ax_comp = plt.subplots(figsize=(14, 7))

    FALLBACK_PALETTE = ["#8C564B", "#E377C2", "#17BECF", "#BCBD22"]
    fallback_iter = iter(FALLBACK_PALETTE)
    fallback_colors = {}

    for column in index_comparison.columns:

        series = index_comparison[column].dropna()

        if len(series) == 0:
            continue

        if column in COLORS:
            color = COLORS[column]
        elif "Min_Variance" in column:
            color = COLORS["Min_Variance"]
        elif "Bipenalized" in column:
            color = COLORS["ERC_Bipenalized"]
        elif "ERC" in column:
            color = COLORS["ERC"]
        elif "Market" in column:
            color = COLORS["Market_Cap"]
        else:
            if column not in fallback_colors:
                fallback_colors[column] = next(fallback_iter, "#7F7F7F")
            color = fallback_colors[column]

        display_label = (
            "Min-Variance bi-pénalisé" if column == "ERC_Bipenalized" else column
        )

        ax_comp.plot(series.index, series.values, linewidth=0.8, label=display_label, color=color)

    ax_comp.set_title("Comparaison des indices — méthodes d'optimisation convexe (Base 100)")
    ax_comp.set_xlabel("Date")
    ax_comp.set_ylabel("Indice — Base 100")
    ax_comp.legend()
    ax_comp.grid(True, alpha=0.3)

    save_chart("53_Index_Comparison_Optimized")
    plt.close()

    # ------------------------------------------------------------
    # 9.6 COMPARAISON DE LA CONCENTRATION HHI
    # ------------------------------------------------------------

    hhi_labels = {
        "Market_Cap": "Pondération par capi",
        "Min_Variance": "Min-Variance",
        "ERC": "ERC",
        w_bipen_series.name: "Min-Variance bi-pénalisé",
    }
    hhi_colors = {
        "Market_Cap": COLORS["Market_Cap"],
        "Min_Variance": COLORS["Min_Variance"],
        "ERC": COLORS["ERC"],
        w_bipen_series.name: COLORS["ERC_Bipenalized"],
    }

    hhi_df = (
        portfolio_stats.set_index("Portfolio")["HHI"]
        .reindex([n for n in hhi_labels])
        .dropna()
    )

    plt.figure(figsize=(9, 6))

    plt.bar(
        [hhi_labels[n] for n in hhi_df.index],
        hhi_df.values,
        color=[hhi_colors[n] for n in hhi_df.index],
    )

    plt.ylabel("HHI")
    plt.xlabel("Méthode de pondération")
    plt.title("Concentration (HHI) selon la méthode de pondération")
    plt.xticks(rotation=20, ha="right")
    plt.grid(axis="y", alpha=0.3)

    save_chart("54_HHI_Comparison")
    plt.close()

    # ============================================================
    # 10. VERIFICATION FINALE DES CONTRAINTES
    # ============================================================

    print("\n" + "-" * 70)
    print("VÉRIFICATION DES CONTRAINTES DE POIDS")
    print("-" * 70)
    print(f"Plafond imposé (Min-Variance / bi-pénalisé) : {w_max:.2%}")
    print(f"Min-Variance : poids max = {w_mv.max():.4%}")
    print(f"ERC (sans plafond) : poids max = {w_erc.max():.4%}")
    print(f"Min-Variance bi-pénalisé : poids max = {w_bipen.max():.4%}")
    print("-" * 70)

    print("\n✓ Optimisation convexe bi-pénalisée terminée avec succès.")

    # ============================================================
    # 11. OBJETS RENVOYES
    # ============================================================

    return {
        "Sigma_clean": Sigma_clean,
        "w_max": w_max,
        "s_sys": s_sys,
        "s_fx": s_fx,
        "weights_min_variance": w_mv_series,
        "weights_erc": w_erc_series,
        "weights_erc_bipenalized": w_bipen_series,
        "risk_contributions_erc": pd.Series(rc_erc, index=tickers),
        "lam_sys": lam_sys,
        "lam_fx": lam_fx,
        "bipenalized_frontier": frontier_df,
        "weights_comparison": weights_comparison,
        "portfolio_stats": portfolio_stats,
        "index_min_variance": index_min_variance,
        "index_erc": index_erc,
        "index_erc_penalized": index_bipenalized,
        "index_comparison": index_comparison,
        "weights_market_cap": pd.Series(w_mc, index=tickers) if w_mc is not None else None,
    }