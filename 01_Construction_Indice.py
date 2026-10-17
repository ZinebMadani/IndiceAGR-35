"""
======================================================================
MODULE 01 - CONSTRUCTION DE L'INDICE BANCAIRE AFRICAIN
======================================================================

Construction de l'indice bancaire africain :

- import des données
- nettoyage
- conversion en devise commune
- traitement spécifique de l'Egypte
- calcul des rendements
- récupération des pondérations
- construction de l'indice
- statistiques descriptives
- tests statistiques
- modèle GARCH
- contributions au rendement
- contributions au risque
- graphiques

IMPORTANT
---------

Les objets produits par run_construction_indice() sont réutilisables
par les modules suivants :

    returns_df
    weights
    cov_matrix
    returns
    risk_table
    fx
    egypt_tickers
    mondays
    non_sunday_mask
    info_country
    etc.
"""

# =====================================================================
# 1. IMPORTS
# =====================================================================

import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PRIMARY_COLOR = "#064E3B"   # SeaGreen

plt.rcParams["axes.prop_cycle"] = plt.cycler(color=[PRIMARY_COLOR])

import scipy.stats as stats

from scipy.stats import (
    norm,
    gaussian_kde,
    jarque_bera,
    shapiro,
    kstest
)

from statsmodels.tsa.stattools import (
    adfuller,
    kpss
)

from statsmodels.graphics.tsaplots import (
    plot_acf,
    plot_pacf
)

from statsmodels.stats.diagnostic import (
    acorr_ljungbox,
    het_arch
)

from arch.unitroot import PhillipsPerron
from arch import arch_model

from openpyxl import load_workbook

from openpyxl.styles import (
    Font,
    Alignment,
    PatternFill,
    Border,
    Side
)

from openpyxl.utils import get_column_letter
from openpyxl.drawing.image import Image as XLImage


# =====================================================================
# 2. IMPORT DU MODULE UTILS
# =====================================================================

import utils


print("\n" + "=" * 70)
print("VERIFICATION DU MODULE UTILS")
print("=" * 70)

print(
    "UTILS UTILISE :",
    utils.__file__
)

print(
    "OUTPUT_FILE EXISTE :",
    hasattr(utils, "OUTPUT_FILE")
)

print(
    "CONTENU OUTPUT_FILE :",
    getattr(
        utils,
        "OUTPUT_FILE",
        "ABSENT"
    )
)


from utils import (
    OUTPUT_FILE,
    CHART_DIR,
    ALPHA,
    EGYPT_NAME,
    charts,
    save_chart,
    safe_remove_directory,
)


# =====================================================================
# 3. PIPELINE COMPLET DE CONSTRUCTION
# =====================================================================

def run_construction_indice():

    """
    Execute l'integralite du pipeline de construction de l'indice
    bancaire africain.

    Les calculs suivent le script d'origine :

    1. Lecture des fichiers
    2. Vérification des informations bancaires
    3. Préparation des cours
    4. Préparation des taux de change
    5. Conversion en devise commune
    6. Récupération des poids
    7. Identification des banques égyptiennes
    8. Suppression des dimanches
    9. Construction de l'indice
    10. Statistiques descriptives
    11. Tests statistiques
    12. GARCH(1,1)
    13. Contributions au rendement
    14. Contributions au risque

    Returns
    -------
    dict
        Ensemble des objets produits et réutilisables par les
        autres modules du projet.
    """

    # =================================================================
    # 4. LECTURE DES FICHIERS
    # =================================================================

    print("\n")
    print("=" * 70)
    print("1. LECTURE DES DONNEES")
    print("=" * 70)

    prices = pd.read_excel(
        "matriceJoursCours.xlsx"
    )

    fx = pd.read_excel(
        "taux.xlsx"
    )

    info = pd.read_excel(
        "banks_info.xlsx"
    )


    # =================================================================
    # 5. VERIFICATION DES INFORMATIONS BANQUES
    # =================================================================

    print("\nVerification des informations bancaires...")

    required_info = [
        "Ticker",
        "Weight"
    ]

    for col in required_info:

        if col not in info.columns:

            raise ValueError(
                f"La colonne '{col}' est absente de "
                "banks_info.xlsx"
            )


    if "Country" not in info.columns:

        raise ValueError(
            "La colonne 'Country' est nécessaire "
            "pour traiter spécifiquement l'Egypte."
        )


    # =================================================================
    # 6. PREPARATION DES COURS
    # =================================================================

    print("\nPréparation des cours...")

    if "Date" not in prices.columns:

        raise ValueError(
            "La colonne 'Date' est absente de "
            "matriceJoursCours.xlsx"
        )


    prices["Date"] = pd.to_datetime(
        prices["Date"],
        errors="coerce"
    )

    prices = prices.dropna(
        subset=["Date"]
    )


    prices = (
        prices
        .drop_duplicates(
            subset=["Date"],
            keep="last"
        )
    )


    prices.set_index(
        "Date",
        inplace=True
    )


    prices = prices.apply(
        pd.to_numeric,
        errors="coerce"
    )


    prices = prices.sort_index()


    # =================================================================
    # 7. VERIFICATION DES COURS
    # =================================================================

    if prices.isna().any().any():

        missing = prices.columns[
            prices.isna().any()
        ].tolist()

        raise ValueError(
            "La matrice des cours contient des valeurs "
            f"manquantes : {missing}"
        )


    # =================================================================
    # 8. PREPARATION DES TAUX DE CHANGE
    # =================================================================

    print(
        "Préparation des taux de change..."
    )


    if "Date" not in fx.columns:

        raise ValueError(
            "La colonne 'Date' est absente de taux.xlsx"
        )


    fx["Date"] = pd.to_datetime(
        fx["Date"],
        errors="coerce"
    )


    fx = fx.dropna(
        subset=["Date"]
    )


    fx = (
        fx
        .drop_duplicates(
            subset=["Date"],
            keep="last"
        )
    )


    fx.set_index(
        "Date",
        inplace=True
    )


    fx = fx.apply(
        pd.to_numeric,
        errors="coerce"
    )


    fx = fx.sort_index()


    # =================================================================
    # 9. VERIFICATION DES TICKERS FX
    # =================================================================

    missing_fx = [
        ticker
        for ticker in prices.columns
        if ticker not in fx.columns
    ]


    if missing_fx:

        raise ValueError(
            "Taux de change absents pour les banques suivantes : "
            + str(missing_fx)
        )


    fx = fx.reindex(
        columns=prices.columns
    )


    # =================================================================
    # 10. ALIGNEMENT INITIAL DES DATES
    # =================================================================

    common_dates = (
        prices.index
        .intersection(
            fx.index
        )
    )


    prices = (
        prices
        .loc[common_dates]
        .sort_index()
    )


    fx = (
        fx
        .loc[common_dates]
        .sort_index()
    )


    # =================================================================
    # 11. VERIFICATION DES TAUX
    # =================================================================

    if fx.isna().any().any():

        missing_fx_dates = fx.columns[
            fx.isna().any()
        ].tolist()

        raise ValueError(
            "La matrice des taux contient des valeurs "
            f"manquantes pour : {missing_fx_dates}"
        )


    if (fx <= 0).any().any():

        raise ValueError(
            "Certains taux de change sont nuls ou négatifs."
        )


    # =================================================================
    # 12. CONVERSION EN DEVISE COMMUNE
    # =================================================================

    print(
        "Conversion des cours en devise commune..."
    )


    prices_common = (
        prices * fx
    )


    # =================================================================
    # 13. RECUPERATION DES POIDS
    # =================================================================

    print(
        "Récupération des pondérations..."
    )


    weights = (
        info
        .set_index("Ticker")["Weight"]
    )


    weights = pd.to_numeric(
        weights,
        errors="coerce"
    )


    weights = weights.reindex(
        prices_common.columns
    )


    if weights.isnull().any():

        missing_tickers = weights[
            weights.isnull()
        ].index.tolist()

        raise ValueError(
            f"Poids absents pour : {missing_tickers}"
        )


    if not np.isclose(
        weights.sum(),
        1
    ):

        raise ValueError(
            f"La somme des poids est "
            f"{weights.sum():.6f} et non 1."
        )


    # =================================================================
    # 14. IDENTIFICATION DES BANQUES EGYPTIENNES
    # =================================================================

    info_country = (
        info
        .set_index("Ticker")["Country"]
    )


    egypt_tickers = []


    for ticker in weights.index:

        if ticker not in info_country.index:

            continue


        country = str(
            info_country.loc[ticker]
        ).strip().upper()


        if country == EGYPT_NAME:

            egypt_tickers.append(
                ticker
            )


    print("\nBanques égyptiennes détectées :")

    for ticker in egypt_tickers:

        print(
            "  ✓",
            ticker
        )


    print(
        f"\nNombre de banques égyptiennes : "
        f"{len(egypt_tickers)}"
    )


    # =================================================================
    # 15. IDENTIFICATION DES DIMANCHES
    # =================================================================

    sunday_mask = (
        prices_common.index.dayofweek == 6
    )


    sunday_dates = (
        prices_common.index[
            sunday_mask
        ]
    )


    print(
        f"\nNombre d'observations du dimanche : "
        f"{len(sunday_dates)}"
    )


    # =================================================================
    # 16. CALCUL DES RENDEMENTS
    # =================================================================

    print(
        "\nCalcul des rendements avec traitement "
        "spécifique de l'Egypte..."
    )


    returns_all = pd.DataFrame(
        index=prices_common.index,
        columns=prices_common.columns,
        dtype=float
    )


    # ---------------------------------------------------------------
    # 16.1 RENDEMENTS STANDARDS
    # ---------------------------------------------------------------

    returns_standard = np.log(
        prices_common /
        prices_common.shift(1)
    )


    returns_all.loc[
        returns_standard.index,
        returns_standard.columns
    ] = returns_standard


    # ---------------------------------------------------------------
    # 16.2 CORRECTION SPECIFIQUE DE L'EGYPTE
    # ---------------------------------------------------------------
    #
    # Marche egyptien : seances du dimanche au jeudi (pas de seance
    # le vendredi ni le samedi). La ligne "dimanche" est neanmoins
    # supprimee de l'indice final (section 17, calendrier commun aux
    # autres marches africains).
    #
    # => Le rendement du LUNDI egyptien est desormais ancre sur le
    #    cours du VENDREDI precedent (et non plus sur le dimanche).
    #    Ancrer sur le dimanche revenait a calculer un rendement
    #    (dimanche -> lundi) dont le point de depart (la ligne du
    #    dimanche) etait ensuite jete : la performance jeudi -> dimanche
    #    du marche egyptien disparaissait alors integralement de
    #    l'indice, sans etre capturee nulle part ailleurs. En ancrant
    #    sur le vendredi (lui-meme deja aligne sur le cours du jeudi,
    #    cf. section "Contraintes de calcul rencontrees" du rapport),
    #    le rendement du lundi capture l'intégralite du mouvement
    #    jeudi -> lundi en une seule observation, sans aucune perte.
    # ---------------------------------------------------------------

    mondays = prices_common.index[
        prices_common.index.dayofweek == 0
    ]


    for monday in mondays:

        friday = (
            monday -
            pd.Timedelta(days=3)
        )


        if friday not in prices_common.index:

            continue


        if friday.dayofweek != 4:

            continue


        for ticker in egypt_tickers:

            friday_price = (
                prices_common
                .loc[
                    friday,
                    ticker
                ]
            )


            monday_price = (
                prices_common
                .loc[
                    monday,
                    ticker
                ]
            )


            if (
                pd.notna(friday_price)
                and
                pd.notna(monday_price)
                and
                friday_price > 0
                and
                monday_price > 0
            ):

                returns_all.loc[
                    monday,
                    ticker
                ] = np.log(
                    monday_price /
                    friday_price
                )


    # =================================================================
    # 17. SUPPRESSION DES DIMANCHES
    # =================================================================

    non_sunday_mask = (
        returns_all.index.dayofweek != 6
    )


    returns_df = (
        returns_all
        .loc[non_sunday_mask]
        .copy()
    )


    # =================================================================
    # 18. SUPPRESSION DES OBSERVATIONS INEXPLOITABLES
    # =================================================================

    returns_df = returns_df.dropna(
        how="all"
    )


    # =================================================================
    # 19. VERIFICATION DES RENDEMENTS
    # =================================================================

    if returns_df.isna().any().any():

        missing_returns = returns_df.columns[
            returns_df.isna().any()
        ].tolist()

        raise ValueError(
            "Des rendements sont manquants pour les banques : "
            + str(missing_returns)
        )


    # =================================================================
    # 20. CONTROLE SPECIFIQUE EGYPTE
    # =================================================================
    #
    # Le controle est desormais aligne sur la nouvelle regle : on
    # verifie que le rendement du lundi stocke dans returns_df
    # correspond bien a ln(P_lundi / P_vendredi), et non plus a
    # ln(P_lundi / P_dimanche).
    # =================================================================

    print(
        "\nContrôle des rendements égyptiens du lundi..."
    )


    egypt_monday_checks = []


    for monday in mondays:

        if monday not in returns_df.index:

            continue


        friday = (
            monday -
            pd.Timedelta(days=3)
        )


        if friday not in prices_common.index:

            continue


        for ticker in egypt_tickers:

            if ticker not in returns_df.columns:

                continue


            friday_price = (
                prices_common
                .loc[
                    friday,
                    ticker
                ]
            )


            monday_price = (
                prices_common
                .loc[
                    monday,
                    ticker
                ]
            )


            calculated = np.log(
                monday_price /
                friday_price
            )


            stored = (
                returns_df
                .loc[
                    monday,
                    ticker
                ]
            )


            egypt_monday_checks.append(
                abs(
                    calculated -
                    stored
                )
            )


    if egypt_monday_checks:

        max_egypt_error = max(
            egypt_monday_checks
        )


        print(
            "Erreur maximale pour les lundis égyptiens :",
            max_egypt_error
        )


        if max_egypt_error < 1e-12:

            print(
                "✓ OK : les lundis égyptiens utilisent bien "
                "le cours du vendredi."
            )

        else:

            print(
                "ATTENTION : problème dans le calcul "
                "des lundis égyptiens."
            )

    from utils import compute_annual_capi_weights, align_annual_weights_daily

    poids_annuels = compute_annual_capi_weights()

    # Extraction des poids annuels uniquement
    if isinstance(poids_annuels, dict):
        poids_annuels = poids_annuels["capi_weights_quarterly"]

        poids_quotidien = align_annual_weights_daily(
            poids_annuels,
            list(returns_df.columns),
            returns_df.index
        )
    # =================================================================
    # 21. CONSTRUCTION DE L'INDICE CAPI (PONDÉRATIONS QUOTIDIENNES)
    # =================================================================

    print(
        "\nConstruction de l'indice CAPI avec pondérations quotidiennes..."
    )

    # Rendement quotidien de l'indice :
    # r_I,t = Σ_i w_i,t * r_i,t

    index_returns = (
        returns_df
        .mul(poids_quotidien)
        .sum(axis=1)
    )


    # Niveau de l'indice Base 100 :
    # I_t = 100 × exp(Σ r_I,t)

    index_capi = (
        100 *
        np.exp(
            index_returns.cumsum()
        )
    )


    index_output = pd.DataFrame({

        "Index_CAPI": index_capi,

        "Index_Return": index_returns

    })


    index_output.index.name = "Date"

    # =================================================================
    # 22. RENDEMENT DE L'INDICE
    # =================================================================

    returns = pd.Series(
        index_returns,
        index=index_returns.index
    ).dropna()


    # =================================================================
    # 23. CONTROLE : AUCUN DIMANCHE
    # =================================================================

    if (
        index_output.index.dayofweek == 6
    ).any():

        raise ValueError(
            "ERREUR : un dimanche est encore présent "
            "dans l'indice."
        )


    print(
        "\n✓ Aucun dimanche n'est présent dans l'indice final."
    )


    # =================================================================
    # 24. STATISTIQUES DESCRIPTIVES
    # =================================================================

    print(
        "Calcul des statistiques descriptives..."
    )


    mu = returns.mean()
    sigma = returns.std()
    variance = returns.var()
    skewness = returns.skew()
    kurtosis = returns.kurtosis()


    descriptive = pd.DataFrame({

        "Statistic": [

            "Number of observations",
            "Mean",
            "Standard deviation",
            "Variance",
            "Skewness",
            "Excess kurtosis",
            "Minimum",
            "Maximum",
            "Median"

        ],

        "Value": [

            len(returns),
            mu,
            sigma,
            variance,
            skewness,
            kurtosis,
            returns.min(),
            returns.max(),
            returns.median()

        ]

    })


    # =================================================================
    # 25. TESTS DE NORMALITE
    # =================================================================

    print(
        "Tests de normalité..."
    )


    jb_stat, jb_pvalue = jarque_bera(
        returns
    )


    shapiro_stat, shapiro_pvalue = shapiro(
        returns
    )


    ks_stat, ks_pvalue = kstest(
        returns,
        "norm",
        args=(
            mu,
            sigma
        )
    )


    normality = pd.DataFrame({

        "Test": [

            "Jarque-Bera",
            "Shapiro-Wilk",
            "Kolmogorov-Smirnov"

        ],

        "Statistic": [

            jb_stat,
            shapiro_stat,
            ks_stat

        ],

        "p-value": [

            jb_pvalue,
            shapiro_pvalue,
            ks_pvalue

        ],

        "Decision": [

            (
                "Reject H0"
                if jb_pvalue < ALPHA
                else "Do not reject H0"
            ),

            (
                "Reject H0"
                if shapiro_pvalue < ALPHA
                else "Do not reject H0"
            ),

            (
                "Reject H0"
                if ks_pvalue < ALPHA
                else "Do not reject H0"
            )

        ],

        "Interpretation": [

            (
                "Returns are not normally distributed."
                if jb_pvalue < ALPHA
                else "No evidence against normality."
            ),

            (
                "Returns are not normally distributed."
                if shapiro_pvalue < ALPHA
                else "No evidence against normality."
            ),

            (
                "Returns differ significantly from "
                "the normal distribution."
                if ks_pvalue < ALPHA
                else "No significant difference from "
                "the normal distribution."
            )

        ]

    })


    # =================================================================
    # 26. TESTS DE STATIONNARITE
    # =================================================================

    print(
        "Tests de stationnarité..."
    )


    adf = adfuller(
        returns
    )


    kpss_result = kpss(
        returns,
        regression="c",
        nlags="auto"
    )


    pp = PhillipsPerron(
        returns
    )


    stationarity = pd.DataFrame({

        "Test": [

            "ADF",
            "KPSS",
            "Phillips-Perron"

        ],

        "Statistic": [

            adf[0],
            kpss_result[0],
            pp.stat

        ],

        "p-value": [

            adf[1],
            kpss_result[1],
            pp.pvalue

        ],

        "Null hypothesis": [

            "Unit root",
            "Stationarity",
            "Unit root"

        ],

        "Decision": [

            (
                "Reject H0"
                if adf[1] < ALPHA
                else "Do not reject H0"
            ),

            (
                "Reject H0"
                if kpss_result[1] < ALPHA
                else "Do not reject H0"
            ),

            (
                "Reject H0"
                if pp.pvalue < ALPHA
                else "Do not reject H0"
            )

        ],

        "Interpretation": [

            (
                "The return series is stationary."
                if adf[1] < ALPHA
                else "Evidence of a unit root."
            ),

            (
                "The return series is non-stationary."
                if kpss_result[1] < ALPHA
                else "The return series is stationary."
            ),

            (
                "The return series is stationary."
                if pp.pvalue < ALPHA
                else "Evidence of a unit root."
            )

        ]

    })


    # =================================================================
    # 27. TEST LJUNG-BOX
    # =================================================================

    print(
        "Test de Ljung-Box..."
    )


    ljung_box = acorr_ljungbox(
        returns,
        lags=[
            10,
            20
        ],
        return_df=True
    )


    ljung_results = pd.DataFrame({

        "Lag": [
            10,
            20
        ],

        "Ljung-Box statistic": [

            ljung_box.loc[
                10,
                "lb_stat"
            ],

            ljung_box.loc[
                20,
                "lb_stat"
            ]

        ],

        "p-value": [

            ljung_box.loc[
                10,
                "lb_pvalue"
            ],

            ljung_box.loc[
                20,
                "lb_pvalue"
            ]

        ],

        "Decision": [

            (
                "Reject H0"
                if ljung_box.loc[
                    10,
                    "lb_pvalue"
                ] < ALPHA
                else "Do not reject H0"
            ),

            (
                "Reject H0"
                if ljung_box.loc[
                    20,
                    "lb_pvalue"
                ] < ALPHA
                else "Do not reject H0"
            )

        ],

        "Interpretation": [

            (
                "Significant autocorrelation detected."
                if ljung_box.loc[
                    10,
                    "lb_pvalue"
                ] < ALPHA
                else "No significant autocorrelation detected."
            ),

            (
                "Significant autocorrelation detected."
                if ljung_box.loc[
                    20,
                    "lb_pvalue"
                ] < ALPHA
                else "No significant autocorrelation detected."
            )

        ]

    })


    # =================================================================
    # 28. TEST ARCH-LM
    # =================================================================

    print(
        "Test ARCH-LM..."
    )


    returns_garch = (
        100 *
        returns
    )


    arch_result = het_arch(
        returns_garch
    )


    arch_stat = arch_result[0]
    arch_pvalue = arch_result[1]


    arch_results = pd.DataFrame({

        "Test": [
            "ARCH-LM"
        ],

        "LM statistic": [
            arch_stat
        ],

        "p-value": [
            arch_pvalue
        ],

        "Decision": [

            (
                "Reject H0"
                if arch_pvalue < ALPHA
                else "Do not reject H0"
            )

        ],

        "Interpretation": [

            (
                "Significant ARCH effects detected: "
                "volatility clustering is present."
                if arch_pvalue < ALPHA
                else
                "No significant ARCH effects detected."
            )

        ]

    })


    # =================================================================
    # 29. GARCH(1,1)
    # =================================================================

    print(
        "Estimation du GARCH(1,1)..."
    )


    garch = arch_model(

        returns_garch,

        mean="Constant",

        vol="GARCH",

        p=1,

        q=1,

        dist="normal"

    )


    garch_result = garch.fit(
        disp="off"
    )


    params = garch_result.params
    pvalues = garch_result.pvalues


    garch_results = pd.DataFrame({

        "Parameter":
            params.index,

        "Coefficient":
            params.values,

        "p-value": [

            pvalues[x]
            for x in params.index

        ]

    })


    garch_results[
        "Significant at 5%"
    ] = (
        garch_results[
            "p-value"
        ] < ALPHA
    )


    # =================================================================
    # 30. INTERPRETATION GARCH
    # =================================================================

    alpha = params.get(
        "alpha[1]",
        np.nan
    )


    beta = params.get(
        "beta[1]",
        np.nan
    )


    omega = params.get(
        "omega",
        np.nan
    )


    persistence = (
        alpha +
        beta
    )


    if persistence < 1:

        persistence_interpretation = (
            "Volatility is persistent but mean-reverting."
        )

    elif np.isclose(
        persistence,
        1
    ):

        persistence_interpretation = (
            "Volatility is highly persistent and "
            "close to an IGARCH process."
        )

    else:

        persistence_interpretation = (
            "The estimated volatility process is "
            "not covariance-stationary."
        )


    garch_summary = pd.DataFrame({

        "Indicator": [

            "Omega",
            "Alpha(1)",
            "Beta(1)",
            "Alpha + Beta",
            "Volatility persistence"

        ],

        "Value": [

            omega,
            alpha,
            beta,
            persistence,
            persistence_interpretation

        ]

    })


    # =================================================================
    # 31. SYNTHESE DES TESTS
    # =================================================================

    summary_tests = pd.DataFrame({

        "Test / Analysis": [

            "Normality - Jarque-Bera",
            "Normality - Shapiro-Wilk",
            "Normality - Kolmogorov-Smirnov",

            "Stationarity - ADF",
            "Stationarity - KPSS",
            "Stationarity - Phillips-Perron",

            "Autocorrelation - Ljung-Box (10)",
            "Autocorrelation - Ljung-Box (20)",

            "ARCH effects - ARCH-LM",

            "GARCH volatility persistence"

        ],

        "p-value": [

            jb_pvalue,
            shapiro_pvalue,
            ks_pvalue,

            adf[1],
            kpss_result[1],
            pp.pvalue,

            ljung_box.loc[
                10,
                "lb_pvalue"
            ],

            ljung_box.loc[
                20,
                "lb_pvalue"
            ],

            arch_pvalue,

            np.nan

        ],

        "Conclusion": [

            (
                "Non-normal"
                if jb_pvalue < ALPHA
                else "Compatible with normality"
            ),

            (
                "Non-normal"
                if shapiro_pvalue < ALPHA
                else "Compatible with normality"
            ),

            (
                "Non-normal"
                if ks_pvalue < ALPHA
                else "Compatible with normality"
            ),

            (
                "Stationary"
                if adf[1] < ALPHA
                else "Non-stationary"
            ),

            (
                "Stationary"
                if kpss_result[1] >= ALPHA
                else "Non-stationary"
            ),

            (
                "Stationary"
                if pp.pvalue < ALPHA
                else "Non-stationary"
            ),

            (
                "Autocorrelation present"
                if ljung_box.loc[
                    10,
                    "lb_pvalue"
                ] < ALPHA
                else "No significant autocorrelation"
            ),

            (
                "Autocorrelation present"
                if ljung_box.loc[
                    20,
                    "lb_pvalue"
                ] < ALPHA
                else "No significant autocorrelation"
            ),

            (
                "ARCH effects present"
                if arch_pvalue < ALPHA
                else "No significant ARCH effects"
            ),

            persistence_interpretation

        ]

    })


    # =================================================================
    # 31.6 UTILITAIRE DE CHAINAGE LOG-ADDITIF
    # =================================================================

    def chain_link_log(
        returns_log: pd.DataFrame,
        weights_schedule: dict,
        rebalance_dates: list,
        base_value: float = 100.0
    ):

        """
        Chainage log-additif avec calendrier de rebalancement.

        Cette fonction n'est PAS utilisée pour la construction
        principale de l'indice.

        Elle est conservée comme utilitaire pour une éventuelle
        introduction future d'un calendrier de rebalancement.
        """

        dates = returns_log.index

        log_level = pd.Series(
            0.0,
            index=dates
        )


        rb_idx = 0


        current_w = weights_schedule[
            rebalance_dates[0]
        ]


        for t in range(
            1,
            len(dates)
        ):

            date = dates[t]


            if (
                rb_idx + 1 <
                len(rebalance_dates)
                and
                date >=
                rebalance_dates[
                    rb_idx + 1
                ]
            ):

                rb_idx += 1

                current_w = weights_schedule[
                    rebalance_dates[
                        rb_idx
                    ]
                ]


            r_t = (
                returns_log
                .loc[date]
                .fillna(0.0)
            )


            port_log_return = (
                current_w *
                r_t
            ).sum()


            log_level.iloc[t] = (
                log_level.iloc[t - 1]
                +
                port_log_return
            )


        return (
            base_value *
            np.exp(log_level)
        )


    # =================================================================
    # 32. COMPOSITION DE L'INDICE
    # =================================================================

    weights_info = info.copy()


    weights_info["Weight"] = pd.to_numeric(
        weights_info["Weight"],
        errors="coerce"
    )


    if weights_info["Weight"].isna().any():

        raise ValueError(
            "Certains poids sont manquants ou "
            "non numériques."
        )


    if not np.isclose(
        weights_info["Weight"].sum(),
        1
    ):

        raise ValueError(
            "La somme des poids est différente de 1."
        )


    weights_info = (
        weights_info
        .sort_values(
            "Weight",
            ascending=False
        )
        .reset_index(
            drop=True
        )
    )


    weights_info[
        "Cumulative Weight"
    ] = (
        weights_info[
            "Weight"
        ].cumsum()
    )


    HHI = np.sum(
        weights_info[
            "Weight"
        ] ** 2
    )


    effective = (
        1 / HHI
    )


    top5_weight = (
        weights_info
        .head(5)["Weight"]
        .sum()
    )


    top10_weight = (
        weights_info
        .head(10)["Weight"]
        .sum()
    )


    # =================================================================
    # 33. ANALYSE PAR PAYS
    # =================================================================

    required_country = [
        "Ticker",
        "Country",
        "Weight"
    ]


    for col in required_country:

        if col not in info.columns:

            raise ValueError(
                f"La colonne '{col}' est absente de "
                "banks_info.xlsx"
            )


    nb_banks = (
        info
        .groupby(
            "Country"
        )["Ticker"]
        .count()
        .reset_index()
    )


    nb_banks.columns = [
        "Country",
        "Number of Banks"
    ]


    country_weights = (
        info
        .groupby(
            "Country"
        )["Weight"]
        .sum()
        .reset_index()
    )


    country_weights = (
        country_weights
        .sort_values(
            "Weight",
            ascending=False
        )
    )


    country_weights[
        "Weight (%)"
    ] = (
        country_weights[
            "Weight"
        ] * 100
    )


    avg_weights = (
        info
        .groupby(
            "Country"
        )["Weight"]
        .mean()
        .reset_index()
    )


    avg_weights.columns = [
        "Country",
        "Average Weight"
    ]


    country_summary = (
        country_weights
        .merge(
            nb_banks,
            on="Country"
        )
        .merge(
            avg_weights,
            on="Country"
        )
    )


    country_summary = (
        country_summary
        .sort_values(
            "Weight",
            ascending=False
        )
        .reset_index(
            drop=True
        )
    )


    country_summary[
        "Cumulative Weight"
    ] = (
        country_summary[
            "Weight"
        ].cumsum()
    )


    # =================================================================
    # 34. CONTRIBUTION AU RENDEMENT PAR BANQUE
    # =================================================================

    print(
        "Calcul des contributions au rendement..."
    )


    missing_tickers = [
        ticker
        for ticker in weights.index
        if ticker not in prices_common.columns
    ]


    if missing_tickers:

        raise ValueError(
            "Les tickers suivants sont absents des cours "
            "en devise commune : "
            + str(missing_tickers)
        )


    prices_contribution = (
        prices_common[
            weights.index
        ].copy()
    )


    contribution_returns = (
        returns_df[
            weights.index
        ].copy()
    )


    poids_quotidien_contribution = poids_quotidien.reindex(
        index=contribution_returns.index,
        columns=contribution_returns.columns
    )

    index_return_contribution = (
        contribution_returns
        .mul(poids_quotidien_contribution)
        .sum(axis=1)
    )

    bank_contribution = (
        contribution_returns
        .mul(poids_quotidien_contribution)
    )


    # =================================================================
    # 35. CONTROLE DES CONTRIBUTIONS
    # =================================================================

    difference = (
        bank_contribution.sum(
            axis=1
        )
        -
        index_return_contribution
    ).abs().max()


    print(
        "\nContrôle contribution rendement :"
    )


    print(
        "Erreur maximale :",
        difference
    )


    if difference > 1e-10:

        print(
            "ATTENTION : contrôle non exact."
        )

    else:

        print(
            "OK : somme des contributions = "
            "rendement de l'indice."
        )


    # =================================================================
    # 36. CONTRIBUTION MOYENNE
    # =================================================================

    mean_contribution = (
        bank_contribution
        .mean()
        .sort_values(
            ascending=False
        )
    )


    # =================================================================
    # 37. CONTRIBUTION CUMULEE
    # =================================================================

    total_contribution = (
        bank_contribution
        .sum()
        .sort_values(
            ascending=False
        )
    )


    # =================================================================
    # 38. CONTRIBUTION EN POURCENTAGE
    # =================================================================

    total_index_log_return = (
        index_return_contribution.sum()
    )


    if abs(
        total_index_log_return
    ) > 1e-12:

        contribution_percentage = (
            total_contribution /
            total_index_log_return
        ) * 100

    else:

        contribution_percentage = pd.Series(
            np.nan,
            index=total_contribution.index
        )


    # =================================================================
    # 39. TABLEAU CONTRIBUTION BANQUES
    # =================================================================

    result = pd.DataFrame({

        "Ticker":
            mean_contribution.index,

        "Average Daily Contribution":
            mean_contribution.values,

        "Total Contribution":
            total_contribution.loc[
                mean_contribution.index
            ].values,

        "Contribution (%)":
            contribution_percentage.loc[
                mean_contribution.index
            ].values

    })


    result = result.merge(
        info,
        on="Ticker",
        how="left"
    )


    columns_order = [

        "Ticker",
        "Country",
        "Weight",
        "Average Daily Contribution",
        "Total Contribution",
        "Contribution (%)"

    ]


    result = result[
        [
            col
            for col in columns_order
            if col in result.columns
        ]
    ]


    result = (
        result
        .sort_values(
            "Average Daily Contribution",
            ascending=False
        )
        .reset_index(
            drop=True
        )
    )


    top10_average = (
        result
        .head(10)[
            "Average Daily Contribution"
        ]
        .sum()
    )


    total_average = (
        result[
            "Average Daily Contribution"
        ].sum()
    )


    if abs(
        total_average
    ) > 1e-12:

        top10_average_share = (
            top10_average /
            total_average
        ) * 100

    else:

        top10_average_share = np.nan


    # =================================================================
    # 40. CONTRIBUTION AU RENDEMENT PAR PAYS
    # =================================================================

    country_map = (
        info
        .set_index(
            "Ticker"
        )["Country"]
    )


    missing_country = [
        ticker
        for ticker in bank_contribution.columns
        if ticker not in country_map.index
    ]


    if missing_country:

        raise ValueError(
            "Pays absents pour les tickers suivants : "
            + str(missing_country)
        )


    country_contribution = (
        bank_contribution
        .rename(
            columns=country_map
        )
    )


    country_daily = (
        country_contribution
        .T
        .groupby(
            level=0
        )
        .sum()
        .T
    )


    country_check = (
        country_daily.sum(
            axis=1
        )
        -
        index_return_contribution
    ).abs().max()


    print(
        "\nContrôle contribution par pays :",
        country_check
    )


    if country_check > 1e-10:

        print(
            "ATTENTION : contrôle par pays non exact."
        )

    else:

        print(
            "OK : somme des contributions pays = "
            "rendement de l'indice."
        )


    country_mean = (
        country_daily
        .mean()
        .sort_values(
            ascending=False
        )
    )


    country_total = (
        country_daily
        .sum()
        .sort_values(
            ascending=False
        )
    )


    total_country_contribution = (
        country_total.sum()
    )


    if abs(
        total_country_contribution
    ) > 1e-12:

        country_share = (
            country_total /
            total_country_contribution
        ) * 100

    else:

        country_share = pd.Series(
            np.nan,
            index=country_total.index
        )


    country_result = pd.DataFrame({

        "Country":
            country_mean.index,

        "Average Daily Contribution":
            country_mean.values,

        "Total Contribution":
            country_total.loc[
                country_mean.index
            ].values,

        "Contribution Share (%)":
            country_share.loc[
                country_mean.index
            ].values

    })


    country_result = (
        country_result
        .sort_values(
            "Average Daily Contribution",
            ascending=False
        )
        .reset_index(
            drop=True
        )
    )


    country_weights_contribution = (
        info
        .groupby(
            "Country"
        )["Weight"]
        .sum()
    )


    country_result[
        "Country Weight"
    ] = (
        country_result[
            "Country"
        ].map(
            country_weights_contribution
        )
    )


    country_result = country_result[
        [
            "Country",
            "Country Weight",
            "Average Daily Contribution",
            "Total Contribution",
            "Contribution Share (%)"
        ]
    ]
    # =================================================================
    # 41. CONTRIBUTION AU RISQUE PAR BANQUE
    # =================================================================

    print("\nCalcul de la contribution au risque...")


    # Rendements utilisés pour la matrice de covariance
    risk_returns = (
        contribution_returns[
            weights.index
        ]
        .copy()
    )

    # =================================================================
    # 42. MATRICE DE COVARIANCE
    # =================================================================

    cov_matrix = (
        risk_returns
        .cov()
    )


    # =================================================================
    # 42.5 POIDS COURANT
    # =================================================================

    weights_current = (
        poids_quotidien
        .iloc[-1]
    )


    # =================================================================
    # 43. VECTEUR DES POIDS
    # =================================================================

    w = (
        weights_current
        .reindex(cov_matrix.columns)
        .fillna(0)
        .values
        .reshape(-1, 1)
    )
    # =================================================================
    # 44. VARIANCE DU PORTEFEUILLE
    # =================================================================

    portfolio_variance = float(
        w.T
        @
        cov_matrix.values
        @
        w
    )


    # =================================================================
    # 45. VOLATILITE QUOTIDIENNE
    # =================================================================

    portfolio_volatility = np.sqrt(
        portfolio_variance
    )


    # =================================================================
    # 46. VOLATILITE ANNUALISEE
    # =================================================================

    annual_volatility = (
        portfolio_volatility
        *
        np.sqrt(252)
    )


    # =================================================================
    # 47. CONTRIBUTION MARGINALE AU RISQUE
    # =================================================================

    marginal_risk = (
        cov_matrix.values @ w
    ) / portfolio_volatility


    # =================================================================
    # 48. CONTRIBUTION TOTALE AU RISQUE
    # =================================================================

    risk_contribution = (
        w *
        marginal_risk
    )


    # =================================================================
    # 49. CONTROLE RISQUE
    # =================================================================

    risk_contribution_sum = (
        risk_contribution.sum()
    )


    risk_check = abs(
        risk_contribution_sum
        -
        portfolio_volatility
    )


    print(
        "Erreur contribution risque :",
        risk_check
    )


    # =================================================================
    # 50. CONTRIBUTION RELATIVE AU RISQUE
    # =================================================================

    relative_contribution = (
        risk_contribution.flatten()
        /
        portfolio_volatility
    ) * 100


    # =================================================================
    # 51. TABLEAU RISQUE
    # =================================================================

    risk_table = pd.DataFrame({

        "Ticker": weights_current.index,

        "Weight": weights_current.reindex(weights_current.index).values,

        "Marginal Risk":
            marginal_risk.flatten(),

        "Risk Contribution":
            risk_contribution.flatten(),

        "Relative Contribution (%)":
            relative_contribution

    })


    risk_table = risk_table.merge(
        info[
            [
                "Ticker",
                "Country"
            ]
        ],
        on="Ticker",
        how="left"
    )


    risk_table = (
        risk_table
        .sort_values(
            "Relative Contribution (%)",
            ascending=False
        )
        .reset_index(
            drop=True
        )
    )


    risk_table = risk_table[
        [
            "Ticker",
            "Country",
            "Weight",
            "Marginal Risk",
            "Risk Contribution",
            "Relative Contribution (%)"
        ]
    ]


    # =================================================================
    # 52. CONCENTRATION DU RISQUE
    # =================================================================

    top5_risk = (
        risk_table
        .head(5)[
            "Relative Contribution (%)"
        ]
        .sum()
    )


    top10_risk = (
        risk_table
        .head(10)[
            "Relative Contribution (%)"
        ]
        .sum()
    )


    corr = (
        risk_table["Weight"]
        .corr(
            risk_table[
                "Relative Contribution (%)"
            ]
        )
    )


    # =================================================================
    # CONTROLES FINAUX
    # =================================================================

    print("\n" + "=" * 70)
    print("CONTROLES FINAUX DE CONSTRUCTION")
    print("=" * 70)

    print(
        "Nombre de banques :",
        len(weights)
    )

    print(
        "Nombre d'observations de rendement :",
        len(returns_df)
    )

    print(
        "Nombre de pays :",
        info["Country"].nunique()
    )

    print(
        "Somme des poids :",
        weights.sum()
    )

    print(
        "Volatilité quotidienne :",
        portfolio_volatility
    )

    print(
        "Volatilité annualisée :",
        annual_volatility
    )

    print(
        "Somme contributions risque :",
        risk_contribution_sum
    )

    print(
        "Erreur contribution risque :",
        risk_check
    )

    print(
        "HHI :",
        HHI
    )

    print(
        "Nombre effectif de banques :",
        effective
    )

    print(
        "Top 5 poids :",
        top5_weight
    )

    print(
        "Top 10 poids :",
        top10_weight
    )

    print(
        "Top 5 contribution risque (%) :",
        top5_risk
    )

    print(
        "Top 10 contribution risque (%) :",
        top10_risk
    )

    print(
        "Corrélation poids / contribution risque :",
        corr
    )

    print("=" * 70)


    # =================================================================
    # 53. OBJETS RETOURNES
    # =================================================================

    construction_results = {

        # -------------------------------------------------------------
        # Données principales
        # -------------------------------------------------------------

        "prices":
            prices,

        "prices_common":
            prices_common,

        "fx":
            fx,

        "info":
            info,

        "info_country":
            info_country,

        # -------------------------------------------------------------
        # Identification Egypte / calendrier
        # -------------------------------------------------------------

        "egypt_tickers":
            egypt_tickers,

        "mondays":
            mondays,

        "sunday_dates":
            sunday_dates,

        "non_sunday_mask":
            non_sunday_mask,

        # -------------------------------------------------------------
        # Rendements / indice
        # -------------------------------------------------------------

        "returns_all":
            returns_all,

        "returns_df":
            returns_df,

        "returns":
            returns,

        "index_returns":
            index_returns,

        "index_capi":
            index_capi,

        "index_output":
            index_output,

        # -------------------------------------------------------------
        # Poids / composition
        # -------------------------------------------------------------

        "weights":
            weights,

        "weights_info":
            weights_info,

        "HHI":
            HHI,

        "effective":
            effective,

        "top5_weight":
            top5_weight,

        "top10_weight":
            top10_weight,

        # -------------------------------------------------------------
        # Statistiques
        # -------------------------------------------------------------

        "mu":
            mu,

        "sigma":
            sigma,

        "variance":
            variance,

        "skewness":
            skewness,

        "kurtosis":
            kurtosis,

        "descriptive":
            descriptive,

        # -------------------------------------------------------------
        # Tests
        # -------------------------------------------------------------

        "normality":
            normality,

        "stationarity":
            stationarity,

        "adf":
            adf,

        "ljung_box":
            ljung_box,

        "ljung_results":
            ljung_results,

        "arch_pvalue":
            arch_pvalue,

        "arch_results":
            arch_results,

        "summary_tests":
            summary_tests,

        # -------------------------------------------------------------
        # GARCH
        # -------------------------------------------------------------

        "garch":
            garch,

        "garch_result":
            garch_result,

        "garch_results":
            garch_results,

        "garch_summary":
            garch_summary,

        "persistence":
            persistence,

        # -------------------------------------------------------------
        # Composition pays
        # -------------------------------------------------------------

        "nb_banks":
            nb_banks,

        "country_weights":
            country_weights,

        "avg_weights":
            avg_weights,

        "country_summary":
            country_summary,

        # -------------------------------------------------------------
        # Contributions rendement
        # -------------------------------------------------------------

        "prices_contribution":
            prices_contribution,

        "contribution_returns":
            contribution_returns,

        "index_return_contribution":
            index_return_contribution,

        "bank_contribution":
            bank_contribution,

        "mean_contribution":
            mean_contribution,

        "total_contribution":
            total_contribution,

        "contribution_percentage":
            contribution_percentage,

        "result":
            result,

        "top10_average":
            top10_average,

        "total_average":
            total_average,

        "top10_average_share":
            top10_average_share,

        # -------------------------------------------------------------
        # Contributions pays
        # -------------------------------------------------------------

        "country_contribution":
            country_contribution,

        "country_daily":
            country_daily,

        "country_mean":
            country_mean,

        "country_total":
            country_total,

        "country_share":
            country_share,

        "country_result":
            country_result,

        # -------------------------------------------------------------
        # Risque
        # -------------------------------------------------------------

        "risk_returns":
            risk_returns,

        "cov_matrix":
            cov_matrix,

        "w":
            w,

        "portfolio_variance":
            portfolio_variance,

        "portfolio_volatility":
            portfolio_volatility,

        "annual_volatility":
            annual_volatility,

        "marginal_risk":
            marginal_risk,

        "risk_contribution":
            risk_contribution,

        "relative_contribution":
            relative_contribution,

        "risk_table":
            risk_table,

        "risk_check":
            risk_check,

        "top5_risk":
            top5_risk,

        "top10_risk":
            top10_risk,

        "corr":
            corr,

        # -------------------------------------------------------------
        # Utilitaire
        # -------------------------------------------------------------

        "chain_link_log":
            chain_link_log

    }


    print("\n✓ Construction de l'indice terminée.")

    # ============================================================
    # 53. CREATION DES GRAPHIQUES
    # ============================================================

    print(
        "\nCréation des graphiques..."
    )


    # ------------------------------------------------------------
    # 53.1 EVOLUTION INDICE
    # ------------------------------------------------------------

    plt.figure(
        figsize=(12, 6)
    )

    plt.plot(
        index_capi.index,
        index_capi.values,
        linewidth=2
    )

    plt.axhline(
        100,
        linestyle="--",
        linewidth=1
    )

    plt.title(
        "Évolution de l'indice bancaire africain — Base 100"
    )

    plt.xlabel(
        "Date"
    )

    plt.ylabel(
        "Indice"
    )

    plt.grid(
        True,
        alpha=0.3
    )

    save_chart(
        "01_Evolution_Indice"
    )


    # ------------------------------------------------------------
    # 53.2 DISTRIBUTION RENDEMENTS
    # ------------------------------------------------------------

    x = np.linspace(
        returns.min(),
        returns.max(),
        500
    )


    kde = gaussian_kde(
        returns,
        bw_method=0.4
    )


    plt.figure(
        figsize=(10, 6)
    )


    plt.hist( 
        returns, 
        bins=35, 
        density=True, 
        alpha=0.6, 
        label="Histogram",
        color="#94A3B8"
    ) 


    plt.plot( 
        x, 
        kde(x), 
        linewidth=2, 
        label="Kernel Density",
        color="#2563EB"
    ) 


    plt.plot( 
        x, 
        norm.pdf( 
            x,
            mu,
            sigma
        ),
        linewidth=2,
        label="Normal Distribution",
        color="#DC2626"
    )


    plt.title(
        "Distribution des rendements de l'indice"
    )

    plt.xlabel(
        "Rendement logarithmique"
    )

    plt.ylabel(
        "Densité"
    )

    plt.legend()

    plt.grid(
        True,
        alpha=0.3
    )

    save_chart(
        "02_Distribution_Rendements"
    )


    # ------------------------------------------------------------
    # 53.3 QQ PLOT
    # ------------------------------------------------------------

    plt.figure(
        figsize=(7, 7)
    )


    stats.probplot(
        returns,
        dist="norm",
        plot=plt
    )


    plt.title(
        "QQ-Plot des rendements"
    )

    plt.grid(
        True,
        alpha=0.3
    )

    save_chart(
        "03_QQ_Plot"
    )


    # ------------------------------------------------------------
    # 53.4 ACF
    # ------------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(10, 5)
    )


    plot_acf(
        returns,
        lags=40,
        ax=ax
    )


    ax.set_title(
        "ACF des rendements"
    )


    ax.grid(
        True,
        alpha=0.3
    )


    save_chart(
        "04_ACF"
    )


    # ------------------------------------------------------------
    # 53.5 PACF
    # ------------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(10, 5)
    )


    plot_pacf(
        returns,
        lags=40,
        method="ywm",
        ax=ax
    )


    ax.set_title(
        "PACF des rendements"
    )


    ax.grid(
        True,
        alpha=0.3
    )


    save_chart(
        "05_PACF"
    )


    # ------------------------------------------------------------
    # 53.6 POIDS DES BANQUES
    # ------------------------------------------------------------

    plt.figure(
        figsize=(14, 6)
    )


    plt.bar(
        weights_info["Ticker"],
        weights_info["Weight"]
    )


    plt.xticks(
        rotation=90
    )


    plt.ylabel(
        "Poids"
    )


    plt.title(
        "Poids des banques dans l'indice"
    )


    plt.grid(
        axis="y",
        alpha=0.3
    )


    save_chart(
        "06_Poids_Banques"
    )

    # ------------------------------------------------------------
    # 53.8 CONCENTRATION CUMULATIVE
    # ------------------------------------------------------------

    plt.figure(
        figsize=(10, 6)
    )


    plt.plot(
        range(
            1,
            len(weights_info) + 1
        ),
        weights_info[
            "Cumulative Weight"
        ],
        marker="o"
    )


    plt.xlabel(
        "Classement des banques"
    )


    plt.ylabel(
        "Poids cumulé"
    )


    plt.title(
        "Concentration cumulative"
    )


    plt.grid(
        True,
        alpha=0.3
    )


    save_chart(
        "08_Concentration_Cumulative"
    )


    # ------------------------------------------------------------
    # 53.9 TOP 10 + OTHERS
    # ------------------------------------------------------------

    pie = (
        weights_info
        .head(10)
        [
            [
                "Ticker",
                "Weight"
            ]
        ]
        .copy()
    )


    others = (
        weights_info
        .iloc[10:]["Weight"]
        .sum()
    )


    pie = pd.concat(
        [
            pie,

            pd.DataFrame({

                "Ticker": [
                    "Others"
                ],

                "Weight": [
                    others
                ]

            })

        ],
        ignore_index=True
    )


    plt.figure(
        figsize=(9, 9)
    )


    colors = [
        "#2D66E1",  # Bleu
        "#DC2626",  # Rouge
        "#16A34A",  # Vert
        "#9333EA",  # Violet
        "#EA580C",  # Orange
        "#0891B2",  # Cyan
        "#CA8A04",  # Jaune foncé
        "#DB2777",  # Rose
        "#4F46E5",  # Indigo
        "#64748B",  # Gris
        "#A3A3A3"   # Others
    ]


    plt.pie(
        pie["Weight"],
        labels=pie["Ticker"],
        autopct=lambda p: f"{p:.1f}%",
        startangle=90,
        colors=colors[:len(pie)]
    )


    plt.title(
        "Répartition des poids — Top 10 + Others"
    )


    save_chart(
        "09_Repartition_Top10"
    )


    # ------------------------------------------------------------
    # 53.10 POIDS PAR PAYS
    # ------------------------------------------------------------

    plt.figure(
        figsize=(12, 6)
    )


    plt.bar(
        country_summary["Country"],
        country_summary["Weight (%)"]
    )


    plt.xticks(
        rotation=45,
        ha="right"
    )


    plt.ylabel(
        "Poids (%)"
    )


    plt.xlabel(
        "Pays"
    )


    plt.title(
        "Poids de chaque pays dans l'indice"
    )


    plt.grid(
        axis="y",
        alpha=0.3
    )


    save_chart(
        "10_Poids_Pays"
    )


    # ------------------------------------------------------------
    # 53.11 COMPOSITION GEOGRAPHIQUE
    # ------------------------------------------------------------

    plt.figure(
        figsize=(9, 9)
    )


    colors = plt.cm.tab20(
        range(len(country_summary))
    )


    plt.pie(
        country_summary["Weight"],
        labels=country_summary["Country"],
        autopct=lambda p: f"{p:.1f}%",
        startangle=90,
        colors=colors
    )


    plt.title(
        "Composition géographique de l'indice"
    )


    save_chart(
        "11_Composition_Geographique"
    )

    # ------------------------------------------------------------
    # 53.15 RENDEMENT QUOTIDIEN
    # ------------------------------------------------------------

    plt.figure(
        figsize=(13, 6)
    )


    plt.plot(
        index_return_contribution.index,
        index_return_contribution.values,
        linewidth=1
    )


    plt.title(
        "Rendement logarithmique quotidien de l'indice"
    )


    plt.xlabel(
        "Date"
    )


    plt.ylabel(
        "Log-Return"
    )


    plt.grid(
        True,
        alpha=0.3
    )


    save_chart(
        "15_Rendement_Quotidien"
    )

    # ------------------------------------------------------------
    # 53.19 CONTRIBUTION CUMULEE DANS LE TEMPS
    # ------------------------------------------------------------

    country_cumulative = (
        country_daily.cumsum()
    )


    plt.figure(
        figsize=(13, 7)
    )


    for country in country_cumulative.columns:

        plt.plot(
            country_cumulative.index,
            country_cumulative[country],
            label=country
        )


    plt.legend(
        bbox_to_anchor=(
            1.02,
            1
        ),
        loc="upper left"
    )


    plt.title(
        "Contribution cumulée au rendement par pays"
    )


    plt.xlabel(
        "Date"
    )


    plt.ylabel(
        "Contribution cumulée"
    )


    plt.grid(
        True,
        alpha=0.3
    )


    save_chart(
        "19_Contribution_Cumulative_Pays"
    )

    # ------------------------------------------------------------
    # 53.22 POIDS VS RISQUE
    # ------------------------------------------------------------

    plt.figure(
        figsize=(9, 7)
    )


    plt.scatter(
        risk_table["Weight"] * 100,
        risk_table[
            "Relative Contribution (%)"
        ]
    )


    for i in range(
        len(risk_table)
    ):

        plt.text(
            risk_table[
                "Weight"
            ].iloc[i] * 100,

            risk_table[
                "Relative Contribution (%)"
            ].iloc[i],

            risk_table[
                "Ticker"
            ].iloc[i],

            fontsize=7
        )


    plt.xlabel(
        "Poids dans l'indice (%)"
    )


    plt.ylabel(
        "Contribution au risque (%)"
    )


    plt.title(
        "Poids vs contribution au risque"
    )


    plt.grid(
        True,
        alpha=0.3
    )


    save_chart(
        "22_Poids_vs_Risque"
    )

    # ============================================================
    # Objets renvoyes pour reutilisation par les autres modules
    # ============================================================
    return {
        "prices": prices,
        "prices_common": prices_common,
        "fx": fx,
        "info": info,
        "weights": weights,
        "weights_current": weights_current,
        "poids_quotidien": poids_quotidien,
        "info_country": info_country,
        "egypt_tickers": egypt_tickers,
        "mondays": mondays,
        "non_sunday_mask": non_sunday_mask,
        "returns_df": returns_df,
        "returns": returns,
        "index_output": index_output,
        "descriptive": descriptive,
        "mu": mu,
        "skewness": skewness,
        "kurtosis": kurtosis,
        "normality": normality,
        "jb_pvalue": jb_pvalue,
        "stationarity": stationarity,
        "adf": adf,
        "ljung_box": ljung_box,
        "ljung_results": ljung_results,
        "arch_pvalue": arch_pvalue,
        "arch_results": arch_results,
        "garch_results": garch_results,
        "garch_summary": garch_summary,
        "persistence_interpretation": persistence_interpretation,
        "summary_tests": summary_tests,
        "weights_info": weights_info,
        "HHI": HHI,
        "effective": effective,
        "top5_weight": top5_weight,
        "top10_weight": top10_weight,
        "country_summary": country_summary,
        "index_return_contribution": index_return_contribution,
        "bank_contribution": bank_contribution,
        "result": result,
        "country_daily": country_daily,
        "country_result": country_result,
        "cov_matrix": cov_matrix,
        "portfolio_variance": portfolio_variance,
        "portfolio_volatility": portfolio_volatility,
        "annual_volatility": annual_volatility,
        "risk_table": risk_table,
        "top5_risk": top5_risk,
        "top10_risk": top10_risk,
        "corr": corr,
    }


# ================================================================
# EXPORT DU CLASSEUR EXCEL
# DASHBOARD + GRAPHIQUES + TOUTES LES FEUILLES
# ================================================================
#
# NOTE (revue de code - nouvelle structure du memoire) :
# - Les "3 methodes de construction alternatives" (Method1/2/3) et
#   leur comparaison (comparison, comparison_log_returns,
#   comparison_simple_returns, comparison_summary) ont ete retirees :
#   ce ne sont pas des indices retenus dans le memoire final. Seul
#   Index_Base100 (construit plus haut via poids_quotidien / CAPI)
#   est produit par ce module.
# - RMT-HRP est abandonne (dendrogramme / HRP retires du memoire) :
#   le parametre rmt_results et tout ce qui en decoulait (rho_hat,
#   eigenvalues, HRP weights, RMT_HRP_Index, feuilles Excel et
#   graphiques associes) ont ete supprimes.
# - Les blocs d'extraction de sr_results (risque systemique) et
#   optim_results (optimisation convexe) etaient auparavant imbriques
#   A L'INTERIEUR du "if rr is not None" : avec rmt_results supprime,
#   ils ne se seraient plus jamais executes. Ils sont desormais
#   independants (chacun teste son propre argument).
# ================================================================

def export_workbook(
    construction_results,
    fx_results=None,
    sr_results=None,
    optim_results=None,
    capi_results=None,
    comparaison_results=None,
):
    """
    Crée le classeur Excel complet à partir des résultats
    de la construction de l'indice et, si disponibles,
    des résultats FX, de risque systémique et d'optimisation convexe.

    Parameters
    ----------
    construction_results : dict
        Résultats de run_construction_indice().

    fx_results : dict or None
        Résultats du module FX (décomposition / attribution).
        Si None, les éléments FX ne sont pas exportés.

    sr_results : dict or None
        Résultats du module de risque systémique.

    optim_results : dict or None
        Résultats du module d'optimisation convexe
        (Index_Min_Variance, Index_ERC, Index_ERC_Bipenalized).

    comparaison_results : dict or None
        Résultats de 06_Comparaison_Indices.run_comparaison_indices()
        (Index_CAPI, Index_Min_Variance, Index_ERC,
        Index_ERC_Penalized) ; comparaison_results["comparison_statistics"]
        est exporté tel quel dans la feuille "Comparaison_Indices_Stats".
    """

    # ============================================================
    # 1. RECUPERATION DES RESULTATS
    # ============================================================

    cr = construction_results
    fr = fx_results
    sr = sr_results
    orr = optim_results
    capr = capi_results
    compr = comparaison_results

    # ============================================================
    # OBJETS PRODUITS PAR LA CONSTRUCTION DE L'INDICE
    # ============================================================

    HHI = cr["HHI"]

    adf = cr["adf"]

    annual_volatility = cr[
        "annual_volatility"
    ]

    arch_pvalue = cr[
        "arch_pvalue"
    ]

    arch_results = cr[
        "arch_results"
    ]

    bank_contribution = cr[
        "bank_contribution"
    ]

    corr = cr[
        "corr"
    ]

    country_daily = cr[
        "country_daily"
    ]

    country_result = cr[
        "country_result"
    ]

    country_summary = cr[
        "country_summary"
    ]

    cov_matrix = cr[
        "cov_matrix"
    ]

    descriptive = cr[
        "descriptive"
    ]

    effective = cr[
        "effective"
    ]

    fx = cr[
        "fx"
    ]

    garch_results = cr[
        "garch_results"
    ]

    garch_summary = cr[
        "garch_summary"
    ]

    index_output = cr[
        "index_output"
    ]

    index_return_contribution = cr[
        "index_return_contribution"
    ]

    jb_pvalue = cr[
        "jb_pvalue"
    ]

    kurtosis = cr[
        "kurtosis"
    ]

    ljung_box = cr[
        "ljung_box"
    ]

    ljung_results = cr[
        "ljung_results"
    ]

    mu = cr[
        "mu"
    ]

    normality = cr[
        "normality"
    ]

    persistence_interpretation = cr[
        "persistence_interpretation"
    ]

    portfolio_variance = cr[
        "portfolio_variance"
    ]

    portfolio_volatility = cr[
        "portfolio_volatility"
    ]

    prices = cr[
        "prices"
    ]

    prices_common = cr[
        "prices_common"
    ]

    result = cr[
        "result"
    ]

    returns = cr[
        "returns"
    ]

    returns_df = cr[
        "returns_df"
    ]

    risk_table = cr[
        "risk_table"
    ]

    skewness = cr[
        "skewness"
    ]

    stationarity = cr[
        "stationarity"
    ]

    summary_tests = cr[
        "summary_tests"
    ]

    top10_risk = cr[
        "top10_risk"
    ]

    top10_weight = cr[
        "top10_weight"
    ]

    top5_risk = cr[
        "top5_risk"
    ]

    top5_weight = cr[
        "top5_weight"
    ]

    weights = cr[
        "weights"
    ]

    weights_info = cr[
        "weights_info"
    ]


    # ============================================================
    # 2. RESULTATS DE RISQUE SYSTEMIQUE
    # ============================================================

    if sr is not None:

        systemic_table = sr["systemic_table"]

        top10_systemic = sr["top10_systemic"]

        system_var_q = sr["system_var_q"]

        system_es_q = sr["system_es_q"]

        systemic_risk_summary = sr["systemic_risk_summary"]


    # ============================================================
    # 3. RESULTATS D'OPTIMISATION CONVEXE
    #    (Index_Min_Variance, Index_ERC, Index_ERC_Bipenalized)
    # ============================================================

    if orr is not None:

        weights_comparison_optim = orr["weights_comparison"]

        portfolio_stats_optim = orr["portfolio_stats"]

        print("\nClés de optimization_results :")
        print(orr.keys())   

        # NOTE (coordination) : la clé réellement produite par
        # 05_Convex_Optimization.py est "bipenalized_frontier"
        # (le modèle unique bi-pénalisé), pas "systemic_frontier".
        systemic_frontier_optim = orr["bipenalized_frontier"]

        index_comparison_optim = orr["index_comparison"]

    if fr is not None:
        print("Clés disponibles dans fr :", fr.keys())
    # ============================================================
    # 4. RESULTATS FX
    # ============================================================

    if fr is not None:

        adjusted_log_return = fr[
            "adjusted_log_return"
        ]

        attribution_index = fr[
            "attribution_index"
        ]

        conditional_risk_summary = fr[
            "conditional_risk_summary"
        ]

        fx_betas = fr[
            "fx_betas"
        ]

        fx_common_factor = fr[
            "fx_common_factor"
        ]

        fx_decomposition_summary = fr[
            "fx_decomposition_summary"
        ]

        mean_fx_systematic_effect = fr[
            "mean_fx_systematic_effect"
        ]

    else:

        adjusted_log_return = None

        attribution_index = None

        conditional_risk_summary = None

        fx_betas = None

        fx_common_factor = None

        fx_decomposition_summary = None

        mean_fx_systematic_effect = None


    # ============================================================
    # 5. CREATION DU FICHIER EXCEL
    # ============================================================

    print(
        "\nCréation du fichier Excel..."
    )


    with pd.ExcelWriter(
        OUTPUT_FILE,
        engine="openpyxl"
    ) as writer:


        # ========================================================
        # 5.1 DONNEES
        # ========================================================

        prices_export = prices.copy()

        prices_export.insert(
            0,
            "Date",
            prices_export.index
        )

        prices_export.to_excel(
            writer,
            sheet_name="Cours_Locaux",
            index=False
        )


        fx_export = fx.copy()

        fx_export.insert(
            0,
            "Date",
            fx_export.index
        )

        fx_export.to_excel(
            writer,
            sheet_name="Taux_Change",
            index=False
        )


        prices_common_export = (
            prices_common.copy()
        )

        prices_common_export.insert(
            0,
            "Date",
            prices_common_export.index
        )

        prices_common_export.to_excel(
            writer,
            sheet_name="Cours_Devise_Commune",
            index=False
        )


        # ========================================================
        # 5.2 RENDEMENTS
        # ========================================================

        returns_output = returns_df.copy()

        returns_output.insert(
            0,
            "Date",
            returns_output.index
        )

        returns_output.to_excel(
            writer,
            sheet_name="Rendements_Banques",
            index=False
        )


        # ========================================================
        # 5.3 INDICE
        # ========================================================

        index_output.to_excel(
            writer,
            sheet_name="Indice"
        )


        # ========================================================
        # 5.4 RENDEMENTS INDICE
        # ========================================================

        returns_index_export = (
            returns
            .to_frame(
                name="Index_Log_Return"
            )
        )

        returns_index_export.index.name = (
            "Date"
        )

        returns_index_export.to_excel(
            writer,
            sheet_name="Rendements_Indice"
        )


        # ========================================================
        # 5.5 STATISTIQUES
        # ========================================================

        descriptive.to_excel(
            writer,
            sheet_name="Descriptives",
            index=False
        )


        normality.to_excel(
            writer,
            sheet_name="Normalite",
            index=False
        )


        stationarity.to_excel(
            writer,
            sheet_name="Stationnarite",
            index=False
        )


        ljung_results.to_excel(
            writer,
            sheet_name="Ljung_Box",
            index=False
        )


        arch_results.to_excel(
            writer,
            sheet_name="ARCH_LM",
            index=False
        )


        garch_results.to_excel(
            writer,
            sheet_name="GARCH_1_1",
            index=False
        )


        garch_summary.to_excel(
            writer,
            sheet_name="GARCH_Summary",
            index=False
        )


        summary_tests.to_excel(
            writer,
            sheet_name="Synthese_Tests",
            index=False
        )


        # ========================================================
        # 5.6 COMPOSITION
        # ========================================================

        weights_info.to_excel(
            writer,
            sheet_name="Composition_Banques",
            index=False
        )


        country_summary.to_excel(
            writer,
            sheet_name="Composition_Pays",
            index=False
        )


        # ========================================================
        # 5.7 CONTRIBUTION BANQUES
        # ========================================================

        result.to_excel(
            writer,
            sheet_name="Contribution_Banques",
            index=False
        )


        bank_contribution_export = (
            bank_contribution.copy()
        )

        bank_contribution_export.insert(
            0,
            "Date",
            bank_contribution_export.index
        )

        bank_contribution_export.to_excel(
            writer,
            sheet_name="Contributions_Quotidiennes",
            index=False
        )


        index_return_export = (
            index_return_contribution
            .to_frame(
                name="Index_Log_Return"
            )
        )

        index_return_export.index.name = (
            "Date"
        )

        index_return_export.to_excel(
            writer,
            sheet_name="Index_Return"
        )


        # ========================================================
        # 5.8 CONTRIBUTION PAYS
        # ========================================================

        country_result.to_excel(
            writer,
            sheet_name="Contribution_Pays",
            index=False
        )


        country_daily_export = (
            country_daily.copy()
        )

        country_daily_export.insert(
            0,
            "Date",
            country_daily_export.index
        )

        country_daily_export.to_excel(
            writer,
            sheet_name="Contributions_Pays_Jour",
            index=False
        )


        # ========================================================
        # 5.9 RISQUE
        # ========================================================

        cov_matrix.to_excel(
            writer,
            sheet_name="Matrice_Covariance"
        )


        risk_table.to_excel(
            writer,
            sheet_name="Risque_Banques",
            index=False
        )


        # ========================================================
        # 5.10 COMPARAISON DES 4 INDICES RETENUS
        # (Index_CAPI / Index_Min_Variance / Index_ERC /
        #  Index_ERC_Penalized), produite par 06_Comparaison_Indices.py
        # ========================================================

        if compr is not None:
            compr["comparison_statistics"].to_excel(
                writer, sheet_name="Comparaison_Indices_Stats"
            )

        # ========================================================
        # 5.10b INDICE PONDERE PAR CAPITALISATION (CAPI)
        # ========================================================
        #
        # NOTE (coordination) : capi_results (capr) etait deja lu en
        # entree de export_workbook mais jamais ecrit dans le
        # classeur. On ajoute ici une feuille minimale de synthese
        # (HHI, Top 5 / Top 10, dernieres ancres) ; les graphiques
        # CAPI (55 a 57) sont deja repris dans chart_order.
        # ========================================================

        if capr is not None:

            if capr.get("capi_summary") is not None:
                capr["capi_summary"].to_excel(
                    writer, sheet_name="Resume_CAPI", index=False
                )

            if capr.get("capi_weights_annual") is not None:
                capr["capi_weights_annual"].to_excel(
                    writer, sheet_name="Poids_CAPI_Annuels"
                )

        # ========================================================
        # 5.11 DECOMPOSITION FX
        # ========================================================

        if fr is not None:


            if fx_decomposition_summary is not None:

                fx_decomposition_summary.to_excel(
                    writer,
                    sheet_name="Decomposition_FX_Banques",
                    index=False
                )


            if (
                conditional_risk_summary is not None
                and conditional_risk_summary.shape[0] > 0
            ):

                conditional_risk_summary.to_excel(
                    writer,
                    sheet_name="Risque_Conditionnel_DCC",
                    index=False
                )


            if fx_betas is not None:

                fx_betas_table = (
                    fx_betas
                    .reset_index()
                )

                fx_betas_table.columns = [
                    "Country",
                    "Beta_Facteur_FX_Commun"
                ]

                fx_betas_table.to_excel(
                    writer,
                    sheet_name="Facteur_FX_Commun_Betas",
                    index=False
                )


            if fx_common_factor is not None:

                fx_common_factor_export = (
                    fx_common_factor
                    .to_frame(
                        name="Facteur_FX_Commun"
                    )
                )

                fx_common_factor_export.index.name = (
                    "Date"
                )

                fx_common_factor_export.to_excel(
                    writer,
                    sheet_name="Facteur_FX_Commun_Serie"
                )


            if attribution_index is not None:

                attribution_index_export = (
                    attribution_index.copy()
                )

                attribution_index_export.index.name = (
                    "Date"
                )

                attribution_index_export.to_excel(
                    writer,
                    sheet_name="Attribution_3_Facteurs"
                )

            if "fx_real_example" in fx_results:
                fx_results["fx_real_example"].to_excel(
                    writer,
                    sheet_name="FX_Real_Example",
                    index=False
                )

        # ========================================================
        # 5.12 RISQUE SYSTEMIQUE
        # ========================================================

        if sr is not None:

            systemic_table.to_excel(
                writer,
                sheet_name="Risque_Systemique",
                index=False
            )

            systemic_risk_summary.to_excel(
                writer,
                    sheet_name="Risque_Systemique_Synthese",
                index=False
            )
        # ========================================================
        # 5.13 OPTIMISATION CONVEXE
        # (Index_Min_Variance / Index_ERC / Index_ERC_Bipenalized)
        # ========================================================

        if orr is not None:

            weights_comparison_optim.to_excel(
                writer,
                sheet_name="Optim_Weights_Comparison",
                index=False
                )

            portfolio_stats_optim.to_excel(
                writer,
                sheet_name="Optim_Portfolio_Stats",
                index=False
            )

            if systemic_frontier_optim is not None:

                systemic_frontier_optim.to_excel(
                    writer,
                    sheet_name="Optim_Systemic_Frontier",
                    index=False
                )

            index_comparison_optim_export = (
                index_comparison_optim.copy()
            )

            index_comparison_optim_export.index.name = "Date"

            index_comparison_optim_export.to_excel(
                writer,
                sheet_name="Optim_Index_Comparison"
            )


    # ============================================================
    # 6. OUVERTURE DU CLASSEUR
    # ============================================================

    wb = load_workbook(
        OUTPUT_FILE
    )


    # ============================================================
    # 7. COULEURS
    # ============================================================

    dark_blue = "17365D"

    blue = "1F4E78"

    light_blue = "D9EAF7"

    green = "E2F0D9"

    red = "FCE4D6"

    grey = "F2F2F2"

    white = "FFFFFF"

    thin_grey = Side(
        style="thin",
        color="D9E1F2"
    )


    # ============================================================
    # 8. STYLE GENERAL
    # ============================================================

    for ws in wb.worksheets:

        ws.sheet_view.showGridLines = False


        # --------------------------------------------------------
        # Freeze panes
        # --------------------------------------------------------

        if ws.max_row > 1:

            ws.freeze_panes = "A2"


        # --------------------------------------------------------
        # AutoFilter
        # --------------------------------------------------------

        if (
            ws.max_row > 1
            and ws.max_column > 1
        ):

            ws.auto_filter.ref = (
                ws.dimensions
            )


        # --------------------------------------------------------
        # Header
        # --------------------------------------------------------

        for cell in ws[1]:

            cell.font = Font(
                bold=True,
                color=white,
                size=11
            )

            cell.fill = PatternFill(
                "solid",
                fgColor=dark_blue
            )

            cell.alignment = Alignment(
                horizontal="center",
                vertical="center",
                wrap_text=True
            )

            cell.border = Border(
                bottom=thin_grey
            )


        ws.row_dimensions[1].height = 30


        # --------------------------------------------------------
        # Alignement
        # --------------------------------------------------------

        for row in ws.iter_rows():

            for cell in row:

                cell.alignment = Alignment(
                    vertical="center"
                )


        # --------------------------------------------------------
        # Largeurs
        # --------------------------------------------------------

        for column in ws.columns:

            max_length = 0

            column_letter = (
                get_column_letter(
                    column[0].column
                )
            )


            for cell in column:

                if cell.value is not None:

                    max_length = max(
                        max_length,
                        len(str(cell.value))
                    )


            ws.column_dimensions[
                column_letter
            ].width = min(
                max(
                    max_length + 2,
                    12
                ),
                45
            )


    # ============================================================
    # 9. FORMATAGE P-VALUES
    # ============================================================

    pvalue_sheets = [

        "Normalite",
        "Stationnarite",
        "Ljung_Box",
        "ARCH_LM",
        "GARCH_1_1",
        "Synthese_Tests"

    ]


    for sheet_name in pvalue_sheets:

        if sheet_name not in wb.sheetnames:

            continue


        ws = wb[sheet_name]

        pvalue_columns = []


        for cell in ws[1]:

            if cell.value is not None:

                name = (
                    str(cell.value)
                    .lower()
                )

                if (
                    "p-value" in name
                    or
                    "pvalue" in name
                ):

                    pvalue_columns.append(
                        cell.column
                    )


        for col_idx in pvalue_columns:

            for row in range(
                2,
                ws.max_row + 1
            ):

                cell = ws.cell(
                    row=row,
                    column=col_idx
                )


                if isinstance(
                    cell.value,
                    (
                        float,
                        int,
                        np.floating
                    )
                ):

                    if cell.value < 0.0001:

                        cell.number_format = (
                            "0.00E+00"
                        )

                    else:

                        cell.number_format = (
                            "0.000000"
                        )


    # ============================================================
    # 10. COLORATION DES DECISIONS
    # ============================================================

    for sheet_name in pvalue_sheets:

        if sheet_name not in wb.sheetnames:

            continue


        ws = wb[sheet_name]

        decision_columns = []


        for cell in ws[1]:

            if cell.value is not None:

                if (
                    "Decision"
                    in str(cell.value)
                ):

                    decision_columns.append(
                        cell.column
                    )


        for col_idx in decision_columns:

            for row in range(
                2,
                ws.max_row + 1
            ):

                cell = ws.cell(
                    row=row,
                    column=col_idx
                )


                if cell.value is None:

                    continue


                value = (
                    str(cell.value)
                    .lower()
                )


                if "reject" in value:

                    cell.fill = PatternFill(
                        "solid",
                        fgColor=red
                    )

                    cell.font = Font(
                        bold=True
                    )


                elif "do not reject" in value:

                    cell.fill = PatternFill(
                        "solid",
                        fgColor=green
                    )

                    cell.font = Font(
                        bold=True
                    )


    # ============================================================
    # 11. FORMATAGE DES POIDS
    # ============================================================

    weight_sheets = [

        "Composition_Banques",
        "Composition_Pays",
        "Contribution_Banques",
        "Contribution_Pays",
        "Risque_Banques"

    ]


    for sheet_name in weight_sheets:

        if sheet_name not in wb.sheetnames:

            continue


        ws = wb[sheet_name]


        for row in ws.iter_rows():

            for cell in row:

                if isinstance(
                    cell.value,
                    (
                        float,
                        int,
                        np.floating
                    )
                ):

                    cell.number_format = (
                        "0.0000"
                    )


    # ============================================================
    # 12. FORMATAGE DES CONTRIBUTIONS %
    # ============================================================

    percentage_columns = {

        "Contribution_Banques": [
            "Contribution (%)"
        ],

        "Contribution_Pays": [
            "Contribution Share (%)"
        ],

        "Risque_Banques": [
            "Relative Contribution (%)"
        ]

    }


    for sheet_name, columns in (
        percentage_columns.items()
    ):

        if sheet_name not in wb.sheetnames:

            continue


        ws = wb[sheet_name]


        headers = {

            str(cell.value):
                cell.column

            for cell in ws[1]

            if cell.value is not None

        }


        for column_name in columns:

            if column_name not in headers:

                continue


            col_idx = headers[
                column_name
            ]


            for row in range(
                2,
                ws.max_row + 1
            ):

                ws.cell(
                    row=row,
                    column=col_idx
                ).number_format = (
                    "0.00"
                )


    # ============================================================
    # 13. DASHBOARD
    # ============================================================

    if "Dashboard" in wb.sheetnames:

        del wb["Dashboard"]


    dashboard = wb.create_sheet(
        "Dashboard",
        0
    )


    dashboard.sheet_view.showGridLines = False


    # ------------------------------------------------------------
    # TITRE
    # ------------------------------------------------------------

    dashboard.merge_cells(
        "A1:H2"
    )


    dashboard["A1"] = (
        "ANALYSE COMPLÈTE DE L'INDICE "
        "BANCAIRE AFRICAIN"
    )


    dashboard["A1"].font = Font(
        bold=True,
        size=20,
        color=white
    )


    dashboard["A1"].fill = PatternFill(
        "solid",
        fgColor=dark_blue
    )


    dashboard["A1"].alignment = Alignment(
        horizontal="center",
        vertical="center"
    )


    # ------------------------------------------------------------
    # SOUS-TITRE
    # ------------------------------------------------------------

    dashboard.merge_cells(
        "A3:H3"
    )


    dashboard["A3"] = (
        "Construction, rendement, risque et exposition au change"
    )


    dashboard["A3"].font = Font(
        italic=True,
        size=11,
        color="666666"
    )


    dashboard["A3"].alignment = Alignment(
        horizontal="center"
    )


    # ============================================================
    # 14. INDICATEURS PRINCIPAUX
    # ============================================================

    dashboard["A5"] = (
        "INDICATEURS PRINCIPAUX"
    )


    dashboard["A5"].font = Font(
        bold=True,
        size=13,
        color=dark_blue
    )


    dashboard["A6"] = "Nombre de banques"
    dashboard["B6"] = len(weights)

    dashboard["C6"] = "Nombre d'observations"
    dashboard["D6"] = len(returns)

    dashboard["E6"] = "Rendement moyen"
    dashboard["F6"] = mu

    dashboard["G6"] = "Volatilité quotidienne"
    dashboard["H6"] = portfolio_volatility


    dashboard["A7"] = "Volatilité annualisée"
    dashboard["B7"] = annual_volatility

    dashboard["C7"] = "Variance quotidienne"
    dashboard["D7"] = portfolio_variance

    dashboard["E7"] = "Skewness"
    dashboard["F7"] = skewness

    dashboard["G7"] = "Excess Kurtosis"
    dashboard["H7"] = kurtosis


    dashboard["A8"] = "HHI"
    dashboard["B8"] = HHI

    dashboard["C8"] = (
        "Nombre effectif de banques"
    )

    dashboard["D8"] = effective

    dashboard["E8"] = "Top 5 poids"
    dashboard["F8"] = top5_weight

    dashboard["G8"] = "Top 10 poids"
    dashboard["H8"] = top10_weight


    for row in range(
        6,
        9
    ):

        for col in range(
            1,
            9
        ):

            cell = dashboard.cell(
                row=row,
                column=col
            )

            cell.border = Border(
                bottom=thin_grey
            )

            cell.alignment = Alignment(
                vertical="center"
            )


    # ============================================================
    # 15. SYNTHESE STATISTIQUE
    # ============================================================

    dashboard["A11"] = (
        "SYNTHÈSE STATISTIQUE"
    )


    dashboard["A11"].font = Font(
        bold=True,
        size=13,
        color=dark_blue
    )


    dashboard["A12"] = "Normalité"

    dashboard["B12"] = (
        "Non normale"
        if jb_pvalue < ALPHA
        else "Compatible avec normalité"
    )


    dashboard["A13"] = "Stationnarité"

    dashboard["B13"] = (
        "Stationnaire"
        if adf[1] < ALPHA
        else "Non stationnaire"
    )


    dashboard["A14"] = "Autocorrélation"

    dashboard["B14"] = (
        "Présente"
        if ljung_box.loc[
            10,
            "lb_pvalue"
        ] < ALPHA
        else "Non significative"
    )


    dashboard["A15"] = "Effets ARCH"

    dashboard["B15"] = (
        "Présents"
        if arch_pvalue < ALPHA
        else "Non significatifs"
    )


    dashboard["A16"] = (
        "Persistance GARCH"
    )

    dashboard["B16"] = (
        persistence_interpretation
    )


    for row in range(
        12,
        17
    ):

        dashboard[
            f"A{row}"
        ].font = Font(
            bold=True
        )

        dashboard[
            f"B{row}"
        ].alignment = Alignment(
            wrap_text=True
        )


    # ============================================================
    # 16. SYNTHESE DU RISQUE
    # ============================================================

    dashboard["D11"] = (
        "SYNTHÈSE DU RISQUE"
    )


    dashboard["D11"].font = Font(
        bold=True,
        size=13,
        color=dark_blue
    )


    dashboard["D12"] = (
        "Top 5 contribution risque"
    )

    dashboard["E12"] = top5_risk


    dashboard["D13"] = (
        "Top 10 contribution risque"
    )

    dashboard["E13"] = top10_risk


    dashboard["D14"] = (
        "Contribution maximale"
    )

    dashboard["E14"] = (
        risk_table[
            "Relative Contribution (%)"
        ].max()
    )


    dashboard["D15"] = (
        "Contribution minimale"
    )

    dashboard["E15"] = (
        risk_table[
            "Relative Contribution (%)"
        ].min()
    )


    dashboard["D16"] = (
        "Corrélation poids / risque"
    )

    dashboard["E16"] = corr


    # ============================================================
    # 17. SYNTHESE DU RISQUE DE CHANGE
    # ============================================================

    dashboard["G11"] = (
        "SYNTHÈSE DU RISQUE DE CHANGE"
    )


    dashboard["G11"].font = Font(
        bold=True,
        size=13,
        color=dark_blue
    )


    if fr is not None and mean_fx_systematic_effect is not None:

        dashboard["G12"] = (
            "Impact moyen du change (facteur commun)"
        )

        dashboard["H12"] = (
            mean_fx_systematic_effect
        )

        dashboard["H12"].number_format = (
            "0.00%"
        )

    else:

        dashboard["G12"] = (
            "Résultats FX non disponibles"
        )

        dashboard.merge_cells(
            "G12:H12"
        )

        dashboard["G12"].alignment = Alignment(
            wrap_text=True
        )


    for row in range(
        12,
        17
    ):

        dashboard[
            f"G{row}"
        ].font = Font(
            bold=True
        )

        dashboard[
            f"H{row}"
        ].alignment = Alignment(
            vertical="center"
        )


    # ============================================================
    # SYNTHESE RISQUE SYSTEMIQUE / OPTIMISATION CONVEXE
    # ============================================================

    dashboard["D19"] = "SYNTHÈSE RISQUE SYSTÉMIQUE"
    dashboard["D19"].font = Font(bold=True, size=13, color=dark_blue)

    dashboard["G19"] = "SYNTHÈSE OPTIMISATION CONVEXE"
    dashboard["G19"].font = Font(bold=True, size=13, color=dark_blue)

    # ------------------------------------------------------------
    # RISQUE SYSTEMIQUE (colonne D / E)
    # ------------------------------------------------------------

    if sr is not None:

        dashboard["D20"] = (
            f"VaR du système ({sr['q']:.0%})"
        )

        dashboard["E20"] = system_var_q

        dashboard["D21"] = f"ES du système ({sr['q']:.0%})"
        dashboard["E21"] = system_es_q

        dashboard["D22"] = "Delta-CoVaR moyen"
        dashboard["E22"] = systemic_table["Delta_CoVaR"].mean()

        dashboard["D23"] = "Banque la plus systémique"
        dashboard["E23"] = systemic_table.iloc[0]["Ticker"]

        dashboard["D24"] = "MES moyen"
        dashboard["E24"] = systemic_table["MES"].mean()

        dashboard["E20"].number_format = "0.0000"
        dashboard["E21"].number_format = "0.0000"
        dashboard["E22"].number_format = "0.0000"
        dashboard["E24"].number_format = "0.0000"

    else:

        dashboard["D20"] = (
            "Résultats de risque systémique non disponibles"
        )

        dashboard.merge_cells("D20:E20")

        dashboard["D20"].alignment = Alignment(wrap_text=True)


    # ------------------------------------------------------------
    # OPTIMISATION CONVEXE (colonne G / H)
    # Index_Min_Variance / Index_ERC / Index_ERC_Bipenalized
    # ------------------------------------------------------------

    if orr is not None:

        def _get_stat(df, portfolio_name, column):

            if (
                df is None
                or portfolio_name not in df["Portfolio"].values
            ):

                return None

            return df.loc[
                df["Portfolio"] == portfolio_name,
                column
            ].values[0]

        dashboard["G20"] = "Lambda retenu (Min-Variance bi-pénalisé)"

        lam_sys_val = orr.get("lam_sys")
        lam_fx_val = orr.get("lam_fx")

        dashboard["H20"] = (
            f"(λ_sys={lam_sys_val:g}, λ_fx={lam_fx_val:g})"
            if lam_sys_val is not None and lam_fx_val is not None
            else None
        )

        dashboard["G21"] = "HHI - Market Cap"
        dashboard["H21"] = _get_stat(
            portfolio_stats_optim, "Market_Cap", "HHI"
        )

        dashboard["G22"] = "HHI - ERC"
        dashboard["H22"] = _get_stat(
            portfolio_stats_optim, "ERC", "HHI"
        )

        dashboard["G23"] = "Volatilité annualisée - Min-Variance"
        dashboard["H23"] = _get_stat(
            portfolio_stats_optim,
            "Min_Variance",
            "Volatilite_Annualisee"
        )

        dashboard["G24"] = "Volatilité annualisée - ERC"
        dashboard["H24"] = _get_stat(
            portfolio_stats_optim, "ERC", "Volatilite_Annualisee"
        )

        dashboard["H20"].number_format = "0.00"
        dashboard["H21"].number_format = "0.0000"
        dashboard["H22"].number_format = "0.0000"
        dashboard["H23"].number_format = "0.00%"
        dashboard["H24"].number_format = "0.00%"

    else:

        dashboard["G20"] = (
            "Résultats d'optimisation convexe non disponibles"
        )

        dashboard.merge_cells("G20:H20")

        dashboard["G20"].alignment = Alignment(wrap_text=True)


    # ------------------------------------------------------------
    # Style commun (gras des libellés, alignement des valeurs)
    # ------------------------------------------------------------

    for row in range(20, 25):

        for col_letter in ["D", "G"]:

            dashboard[f"{col_letter}{row}"].font = Font(bold=True)

        for col_letter in ["E", "H"]:

            dashboard[f"{col_letter}{row}"].alignment = Alignment(
                vertical="center"
            )


    # ============================================================
    # 18. LARGEURS DASHBOARD
    # ============================================================

    dashboard.column_dimensions[
        "A"
    ].width = 30

    dashboard.column_dimensions[
        "B"
    ].width = 28

    dashboard.column_dimensions[
        "C"
    ].width = 30

    dashboard.column_dimensions[
        "D"
    ].width = 28

    dashboard.column_dimensions[
        "E"
    ].width = 25

    dashboard.column_dimensions[
        "F"
    ].width = 22

    dashboard.column_dimensions[
        "G"
    ].width = 30

    dashboard.column_dimensions[
        "H"
    ].width = 22


    for row in range(
        1,
        25
    ):

        dashboard.row_dimensions[
            row
        ].height = 26


    # ============================================================
    # 19. FORMATAGE DASHBOARD
    # ============================================================

    dashboard["B7"].number_format = (
        "0.00%"
    )

    dashboard["F6"].number_format = (
        "0.0000"
    )

    dashboard["H6"].number_format = (
        "0.0000"
    )

    dashboard["B8"].number_format = (
        "0.0000"
    )

    dashboard["D8"].number_format = (
        "0.00"
    )

    dashboard["F8"].number_format = (
        "0.00%"
    )

    dashboard["H8"].number_format = (
        "0.00%"
    )

    dashboard["E12"].number_format = (
        "0.00"
    )

    dashboard["E13"].number_format = (
        "0.00"
    )

    dashboard["E14"].number_format = (
        "0.00"
    )

    dashboard["E15"].number_format = (
        "0.00"
    )

    dashboard["E16"].number_format = (
        "0.000"
    )


    # ============================================================
    # 20. FEUILLE GRAPHIQUES
    # ============================================================

    if "Graphiques" in wb.sheetnames:

        del wb["Graphiques"]


    graphs_ws = wb.create_sheet(
        "Graphiques"
    )


    graphs_ws.sheet_view.showGridLines = False


    graphs_ws.merge_cells(
        "A1:L2"
    )


    graphs_ws["A1"] = (
        "GRAPHIQUES — ANALYSE DE L'INDICE"
    )


    graphs_ws["A1"].font = Font(
        bold=True,
        size=18,
        color=white
    )


    graphs_ws["A1"].fill = PatternFill(
        "solid",
        fgColor=dark_blue
    )


    graphs_ws["A1"].alignment = Alignment(
        horizontal="center",
        vertical="center"
    )


    # ============================================================
    # 21. INSERTION DES GRAPHIQUES
    # ============================================================
    #
    # NOTE (coordination - mise a jour) :
    # - Les graphiques de comparaison des 3 methodes de construction
    #   alternatives (Method1_Prix / Method2_Log) et tous les
    #   graphiques d'allocation HRP / dendrogramme ont ete retires :
    #   ces analyses ne font plus partie de la structure retenue
    #   (module 06 : Index_Base100, Index_CAPI, Index_Min_Variance,
    #   Index_ERC, Index_ERC_Bipenalized).
    # - Le module DCC-GARCH (03_DCC_GARCH_Risk.py) a ete retire du
    #   projet : ses graphiques (ex-32 a 35, "FXShare"/"Correlation
    #   DCC") ne sont plus produits et ont ete enleves de cette
    #   liste.
    # - Cette liste a ete resynchronisee avec les noms/numeros de
    #   graphiques REELLEMENT produits par chaque module (save_chart)
    #   a la date de cette revision : plusieurs entrees etaient
    #   perimees (mauvais numeros/noms issus d'un ancien schema de
    #   numerotation) et les graphiques correspondants n'etaient
    #   donc jamais inseres dans le classeur Excel.
    # ============================================================

    chart_order = [

        # --------------------------------------------------------
        # Construction / composition / risque (module 01)
        # --------------------------------------------------------
        "01_Evolution_Indice",
        "02_Distribution_Rendements",
        "03_QQ_Plot",
        "04_ACF",
        "05_PACF",
        "06_Poids_Banques",
        "08_Concentration_Cumulative",
        "09_Repartition_Top10",
        "10_Poids_Pays",
        "11_Composition_Geographique",
        "15_Rendement_Quotidien",
        "19_Contribution_Cumulative_Pays",
        "22_Poids_vs_Risque",

        # --------------------------------------------------------
        # FX / decomposition (module 02)
        # --------------------------------------------------------
        "29_Evolution_Facteur_FX_Commun",
        "30_Evolution_Effet_Change_Systematique",
        "32_Score_Exposition_FX_Par_Banque",

        # --------------------------------------------------------
        # Nettoyage RMT (module 04_RMT_Cleaning, ex-03_RMT_HRP_Index) :
        # diagnostics uniquement (Sigma_clean alimente l'optimisation
        # convexe, etape 6 ; pas d'indice HRP separe)
        # --------------------------------------------------------
        "36_RMT_Spectrum",
        "37_RMT_Eigenvalues_Cleaning",
        "38_Correlation_Empirical",
        "39_Correlation_RMT_Clean",
        "40_RMT_Correlation_Difference",

        # --------------------------------------------------------
        # Risque systemique (module 04_Systemic_Risk)
        # --------------------------------------------------------
        "41_Top10_Delta_CoVaR",
        "42_Delta_CoVaR_Toutes_Banques",
        "43_VaR_ES_Banques",
        "45_Beta_q_vs_Delta_CoVaR",
        "47_Score_Systemique_Normalise_Par_Banque",

        # --------------------------------------------------------
        # Optimisation convexe bi-penalisee (module 05) :
        # Index_Min_Variance / Index_ERC / Index_ERC_Bipenalized
        # --------------------------------------------------------
        "48_Weights_Comparison_MinVar_ERC_Bipenalized",
        "49_ERC_Risk_Contributions",
        "50_Risk_Frontier_Systemic_Only",
        "51_Risk_Frontier_FX_Only",
        "52_Weights_ERC_vs_Bipenalized",
        "53_Index_Comparison_Optimized",
        "54_HHI_Comparison",

        # --------------------------------------------------------
        # Indice pondere par capitalisation, rebalancement quotidien
        # (module 07)
        # --------------------------------------------------------
        "55_CAPI_Weights_Evolution",
        "57_CAPI_HHI_Evolution",
    ]


    for i, chart_name in enumerate(
        chart_order
    ):

        if chart_name not in charts:

            continue


        row_block = (
            i // 2
        )

        col_block = (
            i % 2
        )


        row = (
            4
            +
            row_block * 30
        )


        col = (
            1
            if col_block == 0
            else 10
        )


        img = XLImage(
            charts[chart_name]
        )


        img.width = 620

        img.height = 360


        cell = (
            get_column_letter(col)
            +
            str(row)
        )


        graphs_ws.add_image(
            img,
            cell
        )

    # ============================================================
    # 22. SAUVEGARDE FINALE
    # ============================================================

    wb.save(
        OUTPUT_FILE
    )

    # ============================================================
    # 22.5 EXPORT DES GRAPHIQUES DANS UN DOSSIER "graphes" (+ ZIP)
    # ============================================================
    #
    # Copie tous les graphiques deja generes vers un dossier
    # persistant "graphes", numerotes dans le meme ordre que
    # l'onglet "Graphiques" du classeur Excel, puis cree une
    # archive ZIP unique pour un telechargement en un seul fichier.
    #
    # ============================================================

    print(
        "\nExport des graphiques dans le dossier 'graphes'..."
    )

    graphes_dir = Path("graphes")

    safe_remove_directory(graphes_dir)

    graphes_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    n_copied = 0

    for i, chart_name in enumerate(chart_order, start=1):

        if chart_name not in charts:
            continue

        source_path = Path(charts[chart_name])

        if not source_path.exists():
            continue

        destination_path = (
            graphes_dir
            / f"{i:02d}_{chart_name}.png"
        )

        shutil.copy2(
            source_path,
            destination_path
        )

        n_copied += 1

    print(
        f"✓ {n_copied} graphiques copies dans "
        f"'{graphes_dir.resolve()}'"
    )

    zip_path = shutil.make_archive(
        base_name="graphes",
        format="zip",
        root_dir=graphes_dir
    )

    print(
        f"✓ Archive ZIP creee : {Path(zip_path).resolve()}"
    )

    # ============================================================
    # 23. SUPPRESSION DES FICHIERS TEMPORAIRES
    # ============================================================

    safe_remove_directory(
        CHART_DIR
    )


    # ============================================================
    # 24. MESSAGE FINAL
    # ============================================================

    print("\n")
    print("=" * 70)
    print("ANALYSE TERMINEE AVEC SUCCES")
    print("=" * 70)


    print(
        f"\nFichier créé : {OUTPUT_FILE}"
    )


    print(
        f"\nNombre de feuilles : "
        f"{len(wb.sheetnames)}"
    )


    print(
        "\nFeuilles du fichier :"
    )


    for sheet in wb.sheetnames:

        print(
            "  ✓",
            sheet
        )


    print(
        "\n✓ Tous les tests statistiques sont dans le fichier."
    )

    print(
        "✓ Tous les tableaux sont dans le fichier."
    )

    print(
        "✓ Tous les graphiques disponibles sont intégrés dans Excel."
    )

    print(
        "✓ Un Dashboard a été créé."
    )

    print(
        "✓ Les décisions des tests sont colorées."
    )

    print(
        "✓ Les dimanches ne sont PAS utilisés comme observations de l'indice."
    )

    print(
        "✓ Le lundi égyptien utilise le cours du vendredi comme référence."
    )

    print(
        "\nFichier final :",
        os.path.abspath(
            OUTPUT_FILE
        )
    )

    print(
        "=" * 70
    )