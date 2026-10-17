"""
05_Systemic_Risk.py
--------------------------------------------------------------
Mesures de risque systemique bancaire :

- VaR et Expected Shortfall (ES) individuels (section 3.1)
- CoVaR et Delta-CoVaR par regression quantile,
  Adrian & Brunnermeier (2016) (section 3.2)
- Marginal Expected Shortfall (MES),
  Acharya, Pedersen, Philippon & Richardson (2017) (section 3.3)
- Score systemique normalise s_i (eq:score_normalise), pret a
  l'emploi comme entree du programme d'optimisation convexe
  bi-penalise (eq:erc_bipenalise)

Ce module reprend EXACTEMENT les memes formules et notations que
le chapitre "Mesures de risque systemique bancaire" (listing 3.1)
pour VaR, ES, CoVaR, Delta-CoVaR et MES : aucun calcul n'est
modifie par rapport au script d'origine (delta_covar,
systemic_risk_scores). Seule l'organisation du code change :
regroupement en une fonction reutilisable run_systemic_risk(),
generation des graphiques via save_chart (partage avec les autres
modules), et un tableau de synthese pret pour l'export Excel.

AJOUT (mise en conformite avec la nouvelle theorie, section II.3
"Mesures de risque systemique", equation eq:score_normalise) :

    Le Delta-CoVaR de chaque banque, deja calcule ci-dessous, est
    normalise entre 0 et 1 :

        s_i = (-Delta_CoVaR_i - min_j(-Delta_CoVaR_j))
              / (max_j(-Delta_CoVaR_j) - min_j(-Delta_CoVaR_j))

    Le signe moins reflete le fait qu'un Delta-CoVaR plus negatif
    signale une contribution plus forte au risque systemique :
    -Delta_CoVaR_i est donc la magnitude de risque systemique de la
    banque i, croissante avec le risque. Le resultat est une
    pd.Series indexee EXACTEMENT comme returns_df.columns (donc
    alignee avec Sigma_clean et avec s^fx, module 02/04), prete a
    servir de vecteur s dans le programme d'optimisation convexe
    bi-penalise :

        w*(lambda_sys, lambda_fx) =
            argmin_w  w' Sigma_clean w
                      + lambda_sys * s'w
                      + lambda_fx  * s^{fx}' w
            s.c.      1'w = 1,  0 <= w <= w_max

RENOMME depuis 04_Systemic_Risk.py : le numero de module 04 est
maintenant pris par 04_RMT_Cleaning.py (nettoyage RMT / Sigma_clean).

Convention de numerotation des graphiques : prefixe "41_" et
suivants (module construction : 01-27b, module FX : 27-31b, module
DCC-GARCH : 32-35, module RMT : 36-40, module risque systemique :
41+).

Ce module ne touche ni au classeur Excel ni au Dashboard : c'est
01_Construction_Indice.py (export_workbook) qui pilote l'export
complet et integre ces resultats.

Objets renvoyes par run_systemic_risk(), directement reutilisables
par export_workbook() et par le module d'optimisation convexe
bi-penalisee (systemic_score = vecteur s pret a l'emploi) :
    systemic_table, top10_systemic, system_var_q, system_es_q,
    systemic_risk_summary, systemic_score.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
PRIMARY_COLOR = "#064E3B"   # SeaGreen

plt.rcParams["axes.prop_cycle"] = plt.cycler(color=[PRIMARY_COLOR])

import statsmodels.formula.api as smf

from utils import save_chart


# ================================================================
# 3.1 VAR ET EXPECTED SHORTFALL INDIVIDUELS
# ================================================================

def value_at_risk(returns: pd.Series, q: float = 0.05) -> float:
    """
    VaR_q : quantile q des rendements.

    VaR_q = inf{x : Pr(R <= x) >= q}
    """

    return returns.quantile(q)


def expected_shortfall(returns: pd.Series, q: float = 0.05) -> float:
    """
    ES_q : perte moyenne au-dela de la VaR_q.

    ES_q = E[R | R <= VaR_q]
    """

    var_q = value_at_risk(returns, q)

    tail = returns[returns <= var_q]

    if len(tail) == 0:
        return np.nan

    return tail.mean()


# ================================================================
# 3.2 COVAR ET DELTA-COVAR (ADRIAN & BRUNNERMEIER, 2016)
# ================================================================

def delta_covar(bank_returns: pd.Series,
                 system_returns: pd.Series,
                 q: float = 0.05):
    """
    bank_returns, system_returns : pd.Series alignees dans le temps.

    Retourne (Delta_CoVaR_q, beta_q) pour la banque consideree, par
    regression quantile (Koenker & Bassett, 1978) du rendement
    systeme sur le rendement de la banque :

        Rj,t = alpha_q + beta_q * Ri,t + eps_q,t

        Delta_CoVaR_q = beta_q_hat * (VaR_i_q - Med_i)
    """

    df = pd.DataFrame({
        "Ri": bank_returns,
        "Rj": system_returns
    }).dropna()

    mod_q = smf.quantreg("Rj ~ Ri", df).fit(q=q)
    mod_med = smf.quantreg("Rj ~ Ri", df).fit(q=0.5)

    var_i_q = df["Ri"].quantile(q)
    med_i = df["Ri"].median()

    covar_stress = (
        mod_q.params["Intercept"] + mod_q.params["Ri"] * var_i_q
    )

    covar_normal = (
        mod_q.params["Intercept"] + mod_q.params["Ri"] * med_i
    )

    return covar_stress - covar_normal, mod_q.params["Ri"]


# ================================================================
# 3.1 + 3.2 + 3.3 TABLEAU COMPLET PAR BANQUE
# ================================================================

def systemic_risk_scores(returns_df: pd.DataFrame,
                          system_returns: pd.Series,
                          q: float = 0.05) -> pd.DataFrame:
    """
    returns_df : DataFrame (T x N), colonnes = les banques.
    system_returns : pd.Series, rendement du systeme (indice agrege).

    Retourne un DataFrame trie par Delta_CoVaR (risque systemique
    decroissant), avec VaR_q, ES_q, MES, beta_q et Delta_CoVaR pour
    chaque banque.
    """

    results = []

    for bank in returns_df.columns:

        bank_returns = returns_df[bank]

        dcovar, beta_q = delta_covar(
            bank_returns,
            system_returns,
            q=q
        )

        var_q = value_at_risk(bank_returns, q=q)
        es_q = expected_shortfall(bank_returns, q=q)

        # ----------------------------------------------------
        # MES : rendement moyen de la banque conditionnel a un
        # jour de detresse du systeme (Rm <= VaR_m_q)
        # ----------------------------------------------------

        aligned = pd.DataFrame({
            "Ri": bank_returns,
            "Rm": system_returns
        }).dropna()

        system_var_q = aligned["Rm"].quantile(q)

        mes = aligned.loc[
            aligned["Rm"] <= system_var_q,
            "Ri"
        ].mean()

        results.append({
            "Ticker": bank,
            "VaR_q": var_q,
            "ES_q": es_q,
            "MES": mes,
            "beta_q": beta_q,
            "Delta_CoVaR": dcovar,
        })

    table = pd.DataFrame(results)

    table = (
        table
        .sort_values("Delta_CoVaR")
        .reset_index(drop=True)
    )

    return table


# ================================================================
# SCORE SYSTEMIQUE NORMALISE s_i (NOUVELLE THEORIE, SECTION II.3,
# EQUATION eq:score_normalise)
# ================================================================

def normalized_systemic_score(delta_covar_by_bank: pd.Series) -> pd.Series:
    """
    Normalise -Delta_CoVaR_i entre 0 et 1 (min-max) :

        s_i = (-Delta_CoVaR_i - min_j(-Delta_CoVaR_j))
              / (max_j(-Delta_CoVaR_j) - min_j(-Delta_CoVaR_j))

    Le signe moins convertit le Delta-CoVaR (generalement negatif,
    d'autant plus negatif que la banque est systemique) en une
    magnitude de risque croissante avec la dangerosite de la banque,
    de sorte que s_i = 1 pour la banque la plus systemique et
    s_i = 0 pour la moins systemique.

    Cas degenere (toutes les banques au meme Delta-CoVaR) : min = max,
    la formule est indeterminee (0/0) ; on retourne alors un score
    neutre de 0.5 pour chaque banque, plutot que de laisser des NaN
    se propager dans l'optimisation convexe en aval.
    """

    magnitude = -delta_covar_by_bank

    magnitude_min = magnitude.min()
    magnitude_max = magnitude.max()

    spread = magnitude_max - magnitude_min

    if spread == 0 or pd.isna(spread):
        return pd.Series(
            0.5,
            index=magnitude.index,
            name="systemic_score",
        )

    score = (magnitude - magnitude_min) / spread

    score.name = "systemic_score"

    return score


# ================================================================
# PIPELINE COMPLET DE RISQUE SYSTEMIQUE
# ================================================================

def run_systemic_risk(returns_df, system_returns, info=None, q=0.05):
    """
    Execute l'ensemble du pipeline de risque systemique bancaire :
    VaR et ES individuels (3.1), CoVaR / Delta-CoVaR par regression
    quantile (3.2), Marginal Expected Shortfall (3.3), et le score
    systemique normalise s_i (section II.3, eq:score_normalise).

    Parameters
    ----------
    returns_df : pandas.DataFrame
        Rendements log des banques (sortie de run_construction_indice(),
        cle "returns_df").
    system_returns : pandas.Series
        Rendement log du systeme / indice agrege (cle "returns" de
        run_construction_indice()).
    info : pandas.DataFrame or None
        Table banks_info (colonnes Ticker, Country, ...) pour
        enrichir le tableau de resultats avec le pays de chaque
        banque.
    q : float
        Niveau de confiance pour la VaR / ES / CoVaR (defaut 5%).

    Returns
    -------
    dict
        Dictionnaire des objets produits, directement reutilisable
        par export_workbook() dans 01_Construction_Indice.py et par
        le module d'optimisation convexe bi-penalisee
        (systemic_score = vecteur s pret a l'emploi, aligne comme
        returns_df.columns).
    """

    print("\n" + "=" * 70)
    print("RISQUE SYSTEMIQUE BANCAIRE")
    print("=" * 70)

    print(f"\nNiveau de confiance : q = {q:.0%}")


    # ============================================================
    # 1. VAR / ES / COVAR / DELTA-COVAR / MES PAR BANQUE
    # ============================================================

    print(
        "\nCalcul de la VaR, ES, Delta-CoVaR et MES par banque..."
    )

    systemic_table = systemic_risk_scores(
        returns_df,
        system_returns,
        q=q
    )

    if info is not None and "Country" in info.columns:

        country_map = info.set_index("Ticker")["Country"]

        systemic_table["Country"] = (
            systemic_table["Ticker"].map(country_map)
        )

        systemic_table = systemic_table[
            [
                "Ticker",
                "Country",
                "VaR_q",
                "ES_q",
                "MES",
                "beta_q",
                "Delta_CoVaR"
            ]
        ]

    systemic_table["Rank_Systemic_Risk"] = range(
        1,
        len(systemic_table) + 1
    )

    print(f"\n{len(systemic_table)} banques analysees.")


    # ============================================================
    # 1.5 SCORE SYSTEMIQUE NORMALISE (s_i, eq:score_normalise)
    # ============================================================

    print(
        "\nCalcul du score systemique normalise (s_i)..."
    )

    _delta_covar_by_bank = systemic_table.set_index("Ticker")["Delta_CoVaR"]

    systemic_score_by_ticker = normalized_systemic_score(
        _delta_covar_by_bank
    )

    systemic_table["Systemic_Score_Normalized"] = (
        systemic_table["Ticker"].map(systemic_score_by_ticker)
    )

    # Vecteur s aligne exactement comme returns_df.columns, pret a
    # l'emploi dans le programme d'optimisation convexe bi-penalise
    # (Sigma_clean et s^fx partagent le meme ordre de tickers).
    systemic_score = systemic_score_by_ticker.reindex(returns_df.columns)
    systemic_score.name = "systemic_score"

    _n_missing_score = systemic_score.isna().sum()

    if _n_missing_score > 0:
        print(
            f"  ! {_n_missing_score} banque(s) sans score s "
            "(absente de systemic_table) : verifier l'alignement des tickers."
        )

    print(
        "Score systemique normalise (s) — min / mediane / max : "
        f"{systemic_score.min():.4f} / "
        f"{systemic_score.median():.4f} / "
        f"{systemic_score.max():.4f}"
    )


    # ============================================================
    # 2. TOP 10 DES CONTRIBUTEURS AU RISQUE SYSTEMIQUE
    # ============================================================

    top10_systemic = systemic_table.head(10).copy()

    print(
        "\nTop 10 des contributeurs au risque systemique "
        "(Delta-CoVaR le plus negatif) :"
    )

    for _, row in top10_systemic.iterrows():

        print(
            f"  {row['Ticker']:<15} "
            f"Delta-CoVaR = {row['Delta_CoVaR']:.6f}"
        )


    # ============================================================
    # 3. VAR ET ES DU SYSTEME (INDICE AGREGE)
    # ============================================================

    system_var_q = value_at_risk(system_returns, q=q)
    system_es_q = expected_shortfall(system_returns, q=q)

    print(f"\nVaR du systeme (q={q:.0%}) : {system_var_q:.6f}")
    print(f"ES du systeme (q={q:.0%})  : {system_es_q:.6f}")


    # ============================================================
    # 4. TABLEAU DE SYNTHESE
    # ============================================================

    systemic_risk_summary = pd.DataFrame({

        "Indicator": [

            f"VaR systeme ({q:.0%})",
            f"ES systeme ({q:.0%})",
            "Delta-CoVaR moyen (toutes banques)",
            "Delta-CoVaR median",
            "Delta-CoVaR min (banque la + risquee)",
            "Delta-CoVaR max (banque la - risquee)",
            "Beta_q moyen",
            "MES moyen",
            "Score systemique moyen (s)",
            "Banque au Delta-CoVaR le plus negatif",
            "Banque au Delta-CoVaR le plus eleve"

        ],

        "Value": [

            system_var_q,
            system_es_q,
            systemic_table["Delta_CoVaR"].mean(),
            systemic_table["Delta_CoVaR"].median(),
            systemic_table["Delta_CoVaR"].min(),
            systemic_table["Delta_CoVaR"].max(),
            systemic_table["beta_q"].mean(),
            systemic_table["MES"].mean(),
            systemic_table["Systemic_Score_Normalized"].mean(),
            systemic_table.iloc[0]["Ticker"],
            systemic_table.iloc[-1]["Ticker"]

        ]

    })


    # ============================================================
    # 5. GRAPHIQUES
    # ============================================================

    print("\nCreation des graphiques de risque systemique...")


    # ------------------------------------------------------------
    # 5.1 TOP 10 DELTA-COVAR
    # ------------------------------------------------------------

    plt.figure(figsize=(12, 6))

    plt.bar(
        top10_systemic["Ticker"],
        top10_systemic["Delta_CoVaR"],
        color="firebrick"
    )

    plt.xticks(rotation=90)
    plt.ylabel("Delta-CoVaR")
    plt.xlabel("Banque")

    plt.title(
        "Top 10 des contributeurs au risque systemique (Delta-CoVaR)"
    )

    plt.grid(axis="y", alpha=0.3)

    save_chart("41_Top10_Delta_CoVaR")


    # ------------------------------------------------------------
    # 5.2 DELTA-COVAR TOUTES LES BANQUES
    # ------------------------------------------------------------

    plt.figure(figsize=(14, 6))

    plt.bar(
        systemic_table["Ticker"],
        systemic_table["Delta_CoVaR"]
    )

    plt.xticks(rotation=90)
    plt.ylabel("Delta-CoVaR")
    plt.xlabel("Banque")

    plt.title(
        "Delta-CoVaR de toutes les banques "
        "(triees par risque systemique)"
    )

    plt.grid(axis="y", alpha=0.3)

    save_chart("42_Delta_CoVaR_Toutes_Banques")


    # ------------------------------------------------------------
    # 5.3 VAR ET ES PAR BANQUE
    # ------------------------------------------------------------

    var_es_sorted = systemic_table.sort_values("ES_q")

    x = np.arange(len(var_es_sorted))
    width = 0.4

    plt.figure(figsize=(14, 6))

    plt.bar( 
        x - width / 2,
        var_es_sorted["VaR_q"],
        width=width,
        label="VaR_q",
        color="#2563EB"
    )

    plt.bar( 
        x + width / 2,
        var_es_sorted["ES_q"],
        width=width,
        label="ES_q",
        color="#DC2626"
    )

    plt.xticks(x, var_es_sorted["Ticker"], rotation=90)
    plt.ylabel("Rendement")
    plt.xlabel("Banque")

    plt.title(
        f"VaR et Expected Shortfall par banque (q={q:.0%})"
    )

    plt.legend()
    plt.grid(axis="y", alpha=0.3)

    save_chart("43_VaR_ES_Banques")


    # ------------------------------------------------------------
    # 5.4 MES PAR BANQUE
    # ------------------------------------------------------------

    mes_sorted = systemic_table.sort_values("MES")

    plt.figure(figsize=(14, 6))

    plt.bar(
        mes_sorted["Ticker"],
        mes_sorted["MES"],
        color="darkorange"
    )

    plt.xticks(rotation=90)
    plt.ylabel("MES")
    plt.xlabel("Banque")

    plt.title(
        "Marginal Expected Shortfall (MES) par banque"
    )

    plt.grid(axis="y", alpha=0.3)

    save_chart("44_MES_Banques")


    # ------------------------------------------------------------
    # 5.5 BETA_Q VS DELTA-COVAR
    # ------------------------------------------------------------

    plt.figure(figsize=(9, 7))

    plt.scatter(
        systemic_table["beta_q"],
        systemic_table["Delta_CoVaR"]
    )

    for _, row in systemic_table.iterrows():

        plt.text(
            row["beta_q"],
            row["Delta_CoVaR"],
            row["Ticker"],
            fontsize=7
        )

    plt.xlabel("Beta_q (sensibilite au systeme)")
    plt.ylabel("Delta-CoVaR")

    plt.title("Beta_q vs Delta-CoVaR")

    plt.grid(True, alpha=0.3)

    save_chart("45_Beta_q_vs_Delta_CoVaR")


    # ------------------------------------------------------------
    # 5.6 DISTRIBUTION DU MES
    # ------------------------------------------------------------

    plt.figure(figsize=(9, 6))

    plt.hist(
        systemic_table["MES"],
        bins=12,
        color="teal",
        alpha=0.7
    )

    plt.xlabel("MES")
    plt.ylabel("Nombre de banques")

    plt.title("Distribution du Marginal Expected Shortfall")

    plt.grid(axis="y", alpha=0.3)

    save_chart("46_Distribution_MES")


    # ------------------------------------------------------------
    # 5.7 SCORE SYSTEMIQUE NORMALISE PAR BANQUE (s)
    # ------------------------------------------------------------

    _score_plot_data = systemic_score.dropna().sort_values()

    plt.figure(
        figsize=(13, max(6, 0.28 * len(_score_plot_data)))
    )

    plt.barh(
        _score_plot_data.index,
        _score_plot_data.values,
        color=PRIMARY_COLOR,
    )

    plt.title(
        "Score de risque systemique normalise (s) par banque"
    )

    plt.xlabel(
        "s_i  (0 = la moins systemique, 1 = la plus systemique)"
    )

    plt.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()

    save_chart("47_Score_Systemique_Normalise_Par_Banque")


    print("\n✓ Risque systemique calcule avec succes.")


    # ============================================================
    # 6. OBJETS RENVOYES
    # ============================================================

    return {

        "q": q,

        "systemic_table": systemic_table,

        "top10_systemic": top10_systemic,

        "system_var_q": system_var_q,

        "system_es_q": system_es_q,

        "systemic_risk_summary": systemic_risk_summary,

        "systemic_score": systemic_score,

    }