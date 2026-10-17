# ============================================================
# 04_RMT_Cleaning.py
# ============================================================
#
# Nettoyage RMT (Random Matrix Theory / theorie des matrices
# aleatoires) de la matrice de correlation empirique des
# rendements bancaires, et construction de la matrice de
# covariance nettoyee Sigma_clean.
#
# Implemente la section "Filtrage RMT de la matrice de
# correlation empirique" (chapitre risque systemique) et
# l'equation (eq:sigma-clean) du memoire :
#
#   Sigma_clean = D . rho_clean . D,   D = diag(sigma_1, ..., sigma_N)
#
# seule la structure de correlation etant nettoyee par RMT, les
# volatilites individuelles restant empiriques.
#
# RENOMME depuis 03_RMT_HRP_Index.py : ce module ne construit
# plus d'indice RMT-HRP separe (classification hierarchique,
# dendrogramme, allocation HRP, indice en niveau) — le memoire
# retient une modelisation unique (section "Optimisation :
# modelisation retenue (modele unique)") ou Sigma_clean entre
# directement, comme terme quadratique w'Sigma_clean w, dans le
# programme d'optimisation convexe bi-penalise (eq:erc_bipenalise) :
#
#   w*(lambda_sys, lambda_fx) =
#       argmin_w  w' Sigma_clean w
#                 + lambda_sys * s'w
#                 + lambda_fx  * s^fx'w
#       s.c.      1'w = 1,  0 <= w <= w_max
#
# Il n'y a donc plus besoin, dans cette partie, ni d'un
# portefeuille a variance minimale construit ici (le portefeuille
# Min-Variance de la section resultats reutilise Sigma_clean
# directement dans le module d'optimisation), ni de la
# classification hierarchique / du dendrogramme HRP.
#
# Convention de numerotation des graphiques : prefixe "36_" et
# suivants (module construction : 01-27b, module FX : 27-31b,
# module DCC-GARCH : 32-35, module RMT : 36+), coherent avec les
# figures citees dans le rapport (36_RMT_Spectrum.png,
# 39_Correlation_RMT_Clean.png).
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
PRIMARY_COLOR = "#064E3B"   # SeaGreen

plt.rcParams["axes.prop_cycle"] = plt.cycler(color=[PRIMARY_COLOR])

from utils import save_chart


# ============================================================
# 1. BORNES DE MARCHENKO-PASTUR
# ============================================================

def marchenko_pastur_bounds(N, T, sigma2=1.0):
    """
    Bornes de la loi de Marchenko-Pastur (eq. section RMT du memoire) :

        lambda_-/+ = sigma2 * (1 -/+ sqrt(q))^2,   q = N / T
    """

    q = N / T

    lam_minus = sigma2 * (1 - np.sqrt(q)) ** 2
    lam_plus = sigma2 * (1 + np.sqrt(q)) ** 2

    return lam_minus, lam_plus


# ============================================================
# 2. FILTRAGE RMT DE LA MATRICE DE CORRELATION
# ============================================================

def rmt_clean_correlation(returns_df):
    """
    Nettoyage RMT (Laloux-Cizeau-Bouchaud-Potters) de la matrice de
    correlation empirique des rendements bancaires : les valeurs
    propres situees dans le bulk de Marchenko-Pastur sont considerees
    comme du bruit d'estimation et remplacees par leur moyenne, la
    premiere valeur propre (facteur de marche) et toute valeur propre
    au-dela de lambda+ etant conservees comme signal.
    """

    if not isinstance(returns_df, pd.DataFrame):
        raise TypeError("returns_df doit être un pandas DataFrame.")

    if returns_df.empty:
        raise ValueError("returns_df est vide.")

    data = returns_df.dropna(how="any").copy()

    if data.shape[1] < 2:
        raise ValueError("Au moins deux banques sont nécessaires.")

    T, N = data.shape

    R = data.values

    stds = R.std(axis=0, ddof=0)

    if np.any(stds == 0):

        bad_columns = data.columns[stds == 0].tolist()

        raise ValueError(
            "Variance nulle pour : " + str(bad_columns)
        )

    Z = (R - R.mean(axis=0)) / stds

    # Corrélation empirique
    rho_hat_array = (Z.T @ Z) / T

    rho_hat = pd.DataFrame(
        rho_hat_array,
        index=data.columns,
        columns=data.columns,
    )

    # --------------------------------------------------------
    # Décomposition spectrale
    # --------------------------------------------------------

    eigvals, eigvecs = np.linalg.eigh(rho_hat_array)

    order = np.argsort(eigvals)[::-1]

    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    # --------------------------------------------------------
    # Bornes MP
    # --------------------------------------------------------

    lam_minus, lam_plus = marchenko_pastur_bounds(N, T)

    # --------------------------------------------------------
    # Facteurs signal (premier facteur = facteur de marché,
    # + toute valeur propre au-dessus de lambda+)
    # --------------------------------------------------------

    is_signal = np.zeros(N, dtype=bool)

    is_signal[0] = True
    is_signal |= (eigvals > lam_plus)

    # --------------------------------------------------------
    # Valeur moyenne du bruit
    # --------------------------------------------------------

    noise_eigvals = eigvals[~is_signal]

    noise_mean = (
        noise_eigvals.mean()
        if len(noise_eigvals) > 0
        else 1.0
    )

    # --------------------------------------------------------
    # Nettoyage
    # --------------------------------------------------------

    eigvals_clean = np.where(is_signal, eigvals, noise_mean)

    eigvals_clean = np.maximum(eigvals_clean, 1e-12)

    # Préserver la trace
    eigvals_clean *= N / eigvals_clean.sum()

    # --------------------------------------------------------
    # Reconstruction
    # --------------------------------------------------------

    rho_clean_array = (eigvecs * eigvals_clean) @ eigvecs.T

    diagonal = np.diag(rho_clean_array)
    diagonal = np.maximum(diagonal, 1e-12)

    d = np.sqrt(diagonal)

    rho_clean_array = rho_clean_array / np.outer(d, d)

    np.fill_diagonal(rho_clean_array, 1.0)

    rho_clean = pd.DataFrame(
        rho_clean_array,
        index=data.columns,
        columns=data.columns,
    )

    # --------------------------------------------------------
    # Tableau des valeurs propres
    # --------------------------------------------------------

    eigenvalues_table = pd.DataFrame({
        "Eigenvalue": eigvals,
        "Cleaned_Eigenvalue": eigvals_clean,
        "MP_Lower_Bound": lam_minus,
        "MP_Upper_Bound": lam_plus,
        "Signal": is_signal,
    })

    eigenvalues_table.index.name = "Eigenvalue_Rank"

    return {
        "rho_hat": rho_hat,
        "rho_clean": rho_clean,
        "eigvals": eigvals,
        "eigvecs": eigvecs,
        "eigvals_clean": eigvals_clean,
        "lambda_minus": lam_minus,
        "lambda_plus": lam_plus,
        "is_signal": is_signal,
        "noise_mean": noise_mean,
        "eigenvalues_table": eigenvalues_table,
        "T": T,
        "N": N,
        "tickers": list(data.columns),
    }


# ============================================================
# 3. MATRICE DE COVARIANCE NETTOYEE (eq:sigma-clean)
# ============================================================

def build_sigma_clean(returns_df, rho_clean):
    """
    Construit Sigma_clean = D . rho_clean . D (eq:sigma-clean du
    memoire) : seule la structure de correlation est nettoyee par
    RMT, les volatilites individuelles (diagonale de D) restant
    empiriques, estimees sur les memes rendements que rho_clean.

    C'est cette matrice — et non rho_clean seule — qui doit etre
    passee comme Sigma_clean au programme d'optimisation convexe
    bi-penalise (eq:erc_bipenalise) : w' Sigma_clean w.

    Parameters
    ----------
    returns_df : pandas.DataFrame
        Rendements bancaires (memes colonnes que rho_clean, dans
        le meme ordre n'est pas requis : reindexation automatique).

    rho_clean : pandas.DataFrame
        Matrice de correlation nettoyee par RMT (sortie de
        rmt_clean_correlation()["rho_clean"]).

    Returns
    -------
    pandas.DataFrame
        Sigma_clean, indexee/colonnee comme rho_clean.
    """

    tickers = list(rho_clean.columns)

    sigma = returns_df[tickers].std(ddof=0)

    D = np.diag(sigma.values)

    Sigma_clean_array = D @ rho_clean.values @ D

    Sigma_clean = pd.DataFrame(
        Sigma_clean_array,
        index=tickers,
        columns=tickers,
    )

    return Sigma_clean


# ============================================================
# 4. GRAPHIQUES DE DIAGNOSTIC RMT
# ============================================================

def plot_rmt_spectrum(eigvals, lambda_plus):

    plt.figure(figsize=(10, 6))

    ranks = np.arange(1, len(eigvals) + 1)

    plt.plot(ranks, eigvals, marker="o", linewidth=1)

    plt.axhline(
        lambda_plus,
        linestyle="--",
        linewidth=2,
        color="darkred",
        label="Borne MP supérieure",
    )

    plt.xlabel("Rang de la valeur propre")
    plt.ylabel("Valeur propre")
    plt.title("Spectre des valeurs propres — Filtrage RMT")
    plt.legend()
    plt.grid(alpha=0.3)

    save_chart("36_RMT_Spectrum")


def plot_eigenvalue_cleaning(eigvals, eigvals_clean, lambda_plus):

    plt.figure(figsize=(11, 6))

    ranks = np.arange(1, len(eigvals) + 1)

    plt.plot(ranks, eigvals, marker="o", label="Empiriques")

    plt.plot(
        ranks,
        eigvals_clean,
        marker="x",
        linestyle="--",
        color="darkorange",
        label="Après nettoyage RMT",
    )

    plt.axhline(
        lambda_plus,
        linestyle=":",
        linewidth=2,
        color="darkred",
        label="Borne MP supérieure",
    )

    plt.xlabel("Rang")
    plt.ylabel("Valeur propre")
    plt.title("Valeurs propres avant et après filtrage RMT")
    plt.legend()
    plt.grid(alpha=0.3)

    save_chart("37_RMT_Eigenvalues_Cleaning")


def plot_correlation_heatmap(matrix, title, chart_name):

    plt.figure(figsize=(11, 9))

    plt.imshow(matrix.values, aspect="auto", interpolation="nearest")

    plt.title(title)

    plt.xticks(
        range(len(matrix.columns)),
        matrix.columns,
        rotation=90,
        fontsize=6,
    )

    plt.yticks(
        range(len(matrix.index)),
        matrix.index,
        fontsize=6,
    )

    plt.colorbar(fraction=0.046, pad=0.04)
    plt.tight_layout()

    save_chart(chart_name)


def plot_correlation_difference(rho_hat, rho_clean):

    difference = rho_hat - rho_clean

    plt.figure(figsize=(11, 9))

    plt.imshow(difference.values, aspect="auto", interpolation="nearest")

    plt.title("Différence : corrélation empirique − corrélation RMT")

    plt.xticks(
        range(len(difference.columns)),
        difference.columns,
        rotation=90,
        fontsize=6,
    )

    plt.yticks(
        range(len(difference.index)),
        difference.index,
        fontsize=6,
    )

    plt.colorbar()
    plt.tight_layout()

    save_chart("40_RMT_Correlation_Difference")


# ============================================================
# 5. FONCTION PRINCIPALE
# ============================================================

def run_rmt_cleaning(returns_df):
    """
    Execute le nettoyage RMT de la matrice de correlation empirique
    des rendements bancaires et construit Sigma_clean, prete a
    l'emploi comme entree du programme d'optimisation convexe
    bi-penalise (eq:erc_bipenalise), aux cotes de s (score
    systemique, module Delta-CoVaR) et s^fx (score d'exposition FX,
    module 02).

    Parameters
    ----------
    returns_df : pandas.DataFrame
        Rendements log par banque (colonnes = tickers, index = dates).
        Utiliser les memes rendements que le reste du pipeline
        (returns_df de run_construction_indice()).

    Returns
    -------
    dict
        {
            "rho_hat": DataFrame,          # correlation empirique
            "rho_clean": DataFrame,        # correlation nettoyee RMT
            "Sigma_clean": DataFrame,      # eq:sigma-clean, entree directe de l'optimisation
            "eigvals": ndarray,
            "eigvecs": ndarray,
            "eigvals_clean": ndarray,
            "lambda_minus": float,
            "lambda_plus": float,
            "is_signal": ndarray,
            "noise_mean": float,
            "eigenvalues_table": DataFrame,
            "n_signal_factors": int,
            "T": int,
            "N": int,
            "tickers": list,
        }
    """

    print("\n" + "=" * 70)
    print("NETTOYAGE RMT DE LA MATRICE DE CORRELATION (Sigma_clean)")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. Filtrage RMT
    # --------------------------------------------------------

    rmt = rmt_clean_correlation(returns_df)

    n_signal_factors = int(rmt["is_signal"].sum())

    q = rmt["N"] / rmt["T"]

    print(f"\nNombre de banques (N) : {rmt['N']}")
    print(f"Nombre d'observations (T) : {rmt['T']}")
    print(f"Ratio q = N/T : {q:.6f}")
    print(f"Borne MP inférieure (lambda-) : {rmt['lambda_minus']:.4f}")
    print(f"Borne MP supérieure (lambda+) : {rmt['lambda_plus']:.4f}")
    print(f"Valeurs propres signal conservées : {n_signal_factors} / {rmt['N']}")
    print(f"Valeur moyenne du bruit : {rmt['noise_mean']:.4f}")

    # --------------------------------------------------------
    # 2. Sigma_clean (eq:sigma-clean)
    # --------------------------------------------------------

    Sigma_clean = build_sigma_clean(returns_df, rmt["rho_clean"])

    print(
        "\nSigma_clean construite : "
        f"{Sigma_clean.shape[0]} x {Sigma_clean.shape[1]}"
    )

    # --------------------------------------------------------
    # 3. Graphiques de diagnostic
    # --------------------------------------------------------

    print("\nGénération des graphiques de diagnostic RMT...")

    plot_rmt_spectrum(rmt["eigvals"], rmt["lambda_plus"])

    plot_eigenvalue_cleaning(
        rmt["eigvals"], rmt["eigvals_clean"], rmt["lambda_plus"]
    )

    plot_correlation_heatmap(
        rmt["rho_hat"],
        "Matrice de corrélation empirique",
        "38_Correlation_Empirical",
    )

    plot_correlation_heatmap(
        rmt["rho_clean"],
        "Matrice de corrélation nettoyée par RMT",
        "39_Correlation_RMT_Clean",
    )

    plot_correlation_difference(rmt["rho_hat"], rmt["rho_clean"])

    print("\n✓ Nettoyage RMT et Sigma_clean terminés.")

    # --------------------------------------------------------
    # Objets renvoyés, réutilisables par le module d'optimisation
    # --------------------------------------------------------

    return {
        "rho_hat": rmt["rho_hat"],
        "rho_clean": rmt["rho_clean"],
        "Sigma_clean": Sigma_clean,
        "eigvals": rmt["eigvals"],
        "eigvecs": rmt["eigvecs"],
        "eigvals_clean": rmt["eigvals_clean"],
        "lambda_minus": rmt["lambda_minus"],
        "lambda_plus": rmt["lambda_plus"],
        "is_signal": rmt["is_signal"],
        "noise_mean": rmt["noise_mean"],
        "eigenvalues_table": rmt["eigenvalues_table"],
        "n_signal_factors": n_signal_factors,
        "T": rmt["T"],
        "N": rmt["N"],
        "tickers": rmt["tickers"],
    }