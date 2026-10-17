# ============================================================
# main.py
# PIPELINE COMPLET - INDICE BANCAIRE AFRICAIN
# ============================================================

import importlib.util
import sys
from pathlib import Path
import pandas as pd
import importlib

import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
# ============================================================
# 1. DOSSIER RACINE DU PROJET
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

sys.path.insert(
    0,
    str(PROJECT_ROOT)
)

# ============================================================
# 2. CHARGEMENT DYNAMIQUE DES MODULES
# ============================================================

def _load_module(
    module_name,
    file_path
):
    """
    Charge dynamiquement un fichier Python.

    Cette méthode est nécessaire car les noms des modules
    commencent par des chiffres.
    """

    file_path = Path(file_path)

    if not file_path.exists():

        raise FileNotFoundError(
            f"\nFichier introuvable :\n"
            f"{file_path}"
        )

    spec = (
        importlib.util
        .spec_from_file_location(
            module_name,
            file_path
        )
    )

    if (
        spec is None
        or spec.loader is None
    ):

        raise ImportError(
            f"Impossible de charger : "
            f"{file_path}"
        )

    module = (
        importlib.util
        .module_from_spec(spec)
    )

    sys.modules[
        module_name
    ] = module

    spec.loader.exec_module(
        module
    )

    return module

# ============================================================
# 3. CHARGEMENT DES MODULES
# ============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "CHARGEMENT DES MODULES"
)

print(
    "=" * 70
)

construction = _load_module(
    "construction_indice",
    PROJECT_ROOT /
    "01_Construction_Indice.py"
)

fx_adjusted = _load_module(
    "fx_adjusted_index",
    PROJECT_ROOT /
    "02_FX_Adjusted_Index.py"
)

rmt_hrp = _load_module(
    "rmt_hrp_index",
    PROJECT_ROOT /
    "03_RMT_HRP_Index.py"
)

systemic_risk = _load_module(
    "systemic_risk",
    PROJECT_ROOT /
    "04_Systemic_Risk.py"
)

convex_optim = _load_module(
    "convex_optimization",
    PROJECT_ROOT /
    "05_Convex_Optimization.py"
)

comparaison_indices = _load_module(
    "comparaison_indices",
    PROJECT_ROOT /
    "06_Comparaison_Indices.py"
)

capi_weights = _load_module(
    "capi_weights",
    PROJECT_ROOT /
    "07_Calcul_Poids_CAPI.py"
)

print(
    "\nModules chargés avec succès."
)


# ============================================================
# 4. FONCTION PRINCIPALE
# ============================================================

def main():

    # ========================================================
    # ETAPE 1
    # CONSTRUCTION DE L'INDICE PRINCIPAL
    # ========================================================

    print(
        "\n\n"
        + "=" * 70
    )

    print(
        "ETAPE 1/7 : CONSTRUCTION DE L'INDICE"
    )

    print(
        "=" * 70
    )

    construction_results = (
        construction
        .run_construction_indice()
    )

    returns_df = construction_results["returns_df"]
    returns = construction_results["returns"]
    poids_quotidien = construction_results["poids_quotidien"]
    fx = construction_results["fx"]
    egypt_tickers = construction_results["egypt_tickers"]
    mondays = construction_results["mondays"]
    non_sunday_mask = construction_results["non_sunday_mask"]
    info_country = construction_results["info_country"]

    # ========================================================
    # VERIFICATION
    # ========================================================

    required_construction = [
        "returns_df",
        "returns",
        "weights",
        "cov_matrix",
        "fx",
        "egypt_tickers",
        "mondays",
        "non_sunday_mask",
        "info_country",
        "info"
    ]

    missing = [
        key
        for key in required_construction
        if key not in construction_results
    ]

    if missing:

        raise KeyError(
            "\nRésultats manquants dans "
            "construction_results :\n"
            + str(missing)
        )

    # ========================================================
    # ETAPE 2
    # INDICE FX-ADJUSTED
    # ========================================================

    print(
        "\n\n"
        + "=" * 70
    )

    print(
        "ETAPE 2/7 : INDICE FX-ADJUSTED"
    )

    print(
        "=" * 70
    )

    try:

        fx_results = fx_adjusted.run_fx_adjusted_index(
            returns_df,
            returns,
            poids_quotidien,
            fx,
            egypt_tickers,
            mondays,
            non_sunday_mask,
            info_country,
        )

    except Exception as e:

        print("\nERREUR MODULE FX :")
        print(type(e))
        print(e)

        raise

    # ========================================================
    # PREPARATION : DOSSIER DE GRAPHIQUES ET INDICES DE
    # COMPARAISON (alimenté au fil des étapes suivantes)
    # ========================================================

    try:

        from utils import CHART_DIR

    except ImportError:

        CHART_DIR = (
            PROJECT_ROOT /
            "temp_charts_indice_africain"
        )

    other_indices = {}

    # ========================================================
    # ETAPE 3
    # INDICE PONDERE PAR CAPITALISATION BOURSIERE (CAPI)
    # ========================================================
    #
    # Placé avant le RMT-HRP afin que Index_CAPI soit lui aussi
    # visible dans les comparaisons des étapes suivantes
    # (RMT-HRP, optimisation convexe, comparaison globale).
    # ========================================================

    print(
        "\n\n"
        + "=" * 70
    )

    print(
        "ETAPE 3/7 : INDICE PONDERE PAR CAPITALISATION (CAPI)"
    )

    print(
        "=" * 70
    )

    try:

        capi_results = (
            capi_weights
            .run_capi_weighted_index(

                returns_df=
                    construction_results[
                        "returns_df"
                    ],

                poids_annuels=None,

                info=
                    construction_results[
                        "info"
                    ],

                other_indices=
                    other_indices
            )
        )

    except FileNotFoundError as e:

        # ----------------------------------------------------
        # Le fichier de capitalisations boursières (utils.CAPI_FILE)
        # est introuvable : on ne bloque pas tout le pipeline
        # pour autant, on continue sans l'indice CAPI et on
        # avertit clairement l'utilisateur.
        # ----------------------------------------------------

        print(
            "\nATTENTION : indice CAPI non calculé "
            f"({e})."
        )

        print(
            "Le pipeline continue sans l'indice CAPI. "
            "Vérifiez utils.CAPI_FILE et le fichier de "
            "capitalisations boursières."
        )

        capi_results = None

    if capi_results is not None:

        required_capi = [
            "capi_weights_annual",
            "capi_weights_daily",
            "index_capi_returns",
            "Index_CAPI",
            "capi_summary",
        ]

        missing_capi = [
            key
            for key in required_capi
            if key not in capi_results
        ]

        if missing_capi:

            raise KeyError(
                "\nRésultats manquants dans capi_results :\n"
                + str(missing_capi)
            )

        print(
            "\nIndice CAPI disponible."
        )

        other_indices["CAPI"] = capi_results["Index_CAPI"]

    # ========================================================
    # ETAPE 4
    # NETTOYAGE RMT (Sigma_clean)
    # ========================================================
    #
    # NOTE (coordination - mise a jour) : 03_RMT_HRP_Index.py a ete
    # renomme en 04_RMT_Cleaning.py et ne construit plus d'indice
    # RMT-HRP separe (dendrogramme / classification hierarchique /
    # allocation HRP retires du memoire, cf. la note dans
    # 01_Construction_Indice.export_workbook). Le module expose
    # desormais une unique fonction run_rmt_cleaning(returns_df),
    # qui ne prend plus ni cov_matrix, ni other_indices, ni
    # chart_dir, ni other_index_weights : elle nettoie la matrice de
    # correlation empirique par RMT et construit directement
    # Sigma_clean (eq:sigma-clean), reutilisee telle quelle comme
    # terme quadratique w' Sigma_clean w dans l'optimisation convexe
    # bi-penalisee (etape 6).
    # ========================================================

    print(
        "\n\n"
        + "=" * 70
    )

    print(
        "ETAPE 4/7 : NETTOYAGE RMT (Sigma_clean)"
    )

    print(
        "=" * 70
    )

    # --------------------------------------------------------
    # Nettoyage RMT de la matrice de correlation + Sigma_clean
    # --------------------------------------------------------

    rmt_results = (
        rmt_hrp
        .run_rmt_cleaning(
            returns_df=
                construction_results[
                    "returns_df"
                ],
        )
    )

    # --------------------------------------------------------
    # Vérification des résultats RMT
    # --------------------------------------------------------

    if rmt_results is None:

        raise RuntimeError(
            "Le module de nettoyage RMT n'a retourné aucun résultat."
        )

    required_rmt = [
        "rho_hat",
        "rho_clean",
        "Sigma_clean",
        "eigvals",
        "eigvecs",
        "eigvals_clean",
        "lambda_minus",
        "lambda_plus",
        "is_signal",
        "noise_mean",
        "eigenvalues_table",
        "n_signal_factors",
        "T",
        "N",
        "tickers",
    ]

    missing_rmt = [
        key
        for key in required_rmt
        if key not in rmt_results
    ]

    if missing_rmt:

        raise KeyError(
            "\nRésultats manquants dans rmt_results :\n"
            + str(missing_rmt)
        )

    print(
        "\nNettoyage RMT et Sigma_clean disponibles."
    )

    # ========================================================
    # ETAPE 5
    # RISQUE SYSTEMIQUE (VAR, ES, COVAR, MES)
    # ========================================================

    print(
        "\n\n"
        + "=" * 70
    )

    print(
        "ETAPE 5/7 : RISQUE SYSTEMIQUE"
    )

    print(
        "=" * 70
    )

    sr_results = (
        systemic_risk
        .run_systemic_risk(

            returns_df=
                construction_results[
                    "returns_df"
                ],

            system_returns=
                construction_results[
                    "returns"
                ],

            info=
                construction_results[
                    "info"
                ],

            q=0.05
        )
    )

    # --------------------------------------------------------
    # Vérification des résultats de risque systémique
    # --------------------------------------------------------

    if sr_results is None:

        raise RuntimeError(
            "Le module de risque systémique n'a retourné "
            "aucun résultat."
        )

    required_sr = [
        "systemic_table",
        "top10_systemic",
        "system_var_q",
        "system_es_q",
        "systemic_risk_summary",
    ]

    missing_sr = [
        key
        for key in required_sr
        if key not in sr_results
    ]

    if missing_sr:

        raise KeyError(
            "\nRésultats manquants dans sr_results :\n"
            + str(missing_sr)
        )

    print(
        "\nRésultats de risque systémique disponibles."
    )

    # ========================================================
    # ETAPE 6
    # OPTIMISATION CONVEXE BI-PENALISEE
    # (MIN-VARIANCE, ERC, MIN-VARIANCE BI-PENALISE)
    # ========================================================
    #
    # NOTE (coordination - mise a jour) : 05_Convex_Optimization.py
    # implemente desormais le "modele unique" bi-penalise
    # (eq:erc_bipenalise) : w' Sigma_clean w + lambda_sys*s'w +
    # lambda_fx*s^fx'w. Il ne prend plus ni cov_matrix, ni
    # rho_clean, ni systemic_table, ni lam_erc_penalized, mais :
    #   - Sigma_clean   (module 04_RMT_Cleaning, rmt_results)
    #   - systemic_score (module 04_Systemic_Risk, sr_results)
    #   - fx_exposure_score (module 02_FX_Adjusted_Index, fx_results)
    #   - lam_sys / lam_fx (les deux coefficients du modele unique,
    #     au lieu de l'ancien lam_erc_penalized).
    # ========================================================

    print(
        "\n\n"
        + "=" * 70
    )

    print(
        "ETAPE 6/7 : OPTIMISATION CONVEXE"
    )

    print(
        "=" * 70
    )

    optim_results = (
        convex_optim
        .run_convex_optimization(

            returns_df=
                construction_results[
                    "returns_df"
                ],

            Sigma_clean=
                rmt_results[
                    "Sigma_clean"
                ],

            systemic_score=
                sr_results[
                    "systemic_score"
                ],

            fx_exposure_score=
                fx_results[
                    "fx_exposure_score"
                ],

            info=
                construction_results[
                    "info"
                ],

            w_max=0.15,

            lam_sys=1.0,

            lam_fx=1.0,

            other_indices=
                other_indices,

            market_cap_weights=construction_results["weights_info"]
                       .set_index("Ticker")["Weight"],
        )
    )

    # --------------------------------------------------------
    # Vérification des résultats d'optimisation convexe
    # --------------------------------------------------------

    if optim_results is None:

        raise RuntimeError(
            "Le module d'optimisation convexe n'a retourné "
            "aucun résultat."
        )

    required_optim = [
        "weights_min_variance",
        "weights_erc",
        "weights_comparison",
        "portfolio_stats",
        "index_comparison",
    ]

    missing_optim = [
        key
        for key in required_optim
        if key not in optim_results
    ]

    if missing_optim:

        raise KeyError(
            "\nRésultats manquants dans optim_results :\n"
            + str(missing_optim)
        )

    print(
        "\nRésultats d'optimisation convexe disponibles."
    )

    # ========================================================
    # ETAPE 7
    # COMPARAISON GLOBALE DES METHODES DE CONSTRUCTION
    # ========================================================
    #
    # Regroupe et compare, sur leur période commune, les 4 indices
    # retenus par le pipeline (cf. comparaison_indices.TARGET_INDICES) :
    # CAPI, Min-Variance, ERC, Min-Variance bi-pénalisé.
    # ========================================================

    print(
        "\n\n"
        + "=" * 70
    )

    print(
        "ETAPE 7/7 : COMPARAISON GLOBALE DES METHODES"
    )

    print(
        "=" * 70
    )

    result_sources = [
        ("construction_results", construction_results),
        ("fx_results", fx_results),
        ("rmt_results", rmt_results),
        ("optim_results", optim_results),
    ]

    if capi_results is not None:

        result_sources.append(
            ("capi_results", capi_results)
        )

    comparaison_results = (
        comparaison_indices
        .run_comparaison_indices(
            result_sources=result_sources,
            chart_dir=CHART_DIR,
            rebase_100=True,
        )
    )

    if comparaison_results is not None:

        # ----------------------------------------------------
        # Le module 06 sauvegarde son graphique directement
        # (fig.savefig), en dehors du registre partagé
        # utils.charts. On l'enregistre ici explicitement pour
        # qu'il soit inséré dans le classeur Excel comme tous
        # les autres graphiques du projet.
        # ----------------------------------------------------

        from utils import register_chart

        register_chart(
            "47_Comparaison_Methodes",
            comparaison_results["chart_path"]
        )

        print(
            "\n✓ Comparaison globale des méthodes disponible "
            f"({len(comparaison_results['detected_indices'])}"
            f"/{len(comparaison_indices.TARGET_INDICES)} "
            "indices détectés)."
        )

        if comparaison_results["missing_indices"]:

            print(
                "  Indices non détectés : "
                + ", ".join(
                    comparaison_results["missing_indices"]
                )
            )

    else:

        print(
            "\nATTENTION : comparaison globale non disponible "
            "(aucun indice détecté)."
        )

    # ========================================================
    # EXPORT EXCEL
    # ========================================================

    print(
        "\n\n"
        + "=" * 70
    )

    print(
        "EXPORT DU CLASSEUR EXCEL COMPLET"
    )

    print(
        "=" * 70
    )

    # --------------------------------------------------------
    # NOTE (coordination - mise a jour) : le module DCC-GARCH
    # (03_DCC_GARCH_Risk.py) a été retiré du projet (plus utilisé
    # dans la dernière version du rapport) : plus d'import, plus de
    # calcul dcc_risk_results, et le paramètre correspondant n'est
    # plus passé à export_workbook().
    #
    # NOTE (coordination) : export_workbook() n'a pas de paramètre
    # rmt_results (le nettoyage RMT / Sigma_clean n'a pas sa propre
    # feuille/section Excel dédiée : il alimente uniquement
    # l'optimisation convexe, étape 6). On appelle avec des
    # arguments nommés, qui correspondent à la signature actuelle
    # de export_workbook().
    # --------------------------------------------------------

    construction.export_workbook(
        construction_results=construction_results,
        fx_results=fx_results,
        sr_results=sr_results,
        optim_results=optim_results,
        capi_results=capi_results,
        comparaison_results=comparaison_results,
    )

    print(
        "\n✓ Risque systémique, optimisation convexe bi-pénalisée, "
        "indice CAPI et comparaison globale exportés dans Excel "
        "(le nettoyage RMT / Sigma_clean alimente l'optimisation "
        "convexe mais n'a plus de feuille Excel dédiée)."
    )

    # ========================================================
    # MESSAGE FINAL
    # ========================================================

    print(
        "\n\n"
        + "=" * 70
    )

    print(
        "PIPELINE TERMINE"
    )

    print(
        "=" * 70
    )

    print(
        "\nLes sept étapes ont été exécutées :"
    )

    print(
        "  [1] Construction de l'indice Market Cap "
        "(Base 100)"
    )

    print(
        "  [2] Construction de l'indice FX-Adjusted"
    )

    print(
        "  [3] Construction de l'indice pondéré par "
        "capitalisation (CAPI)"
    )

    print(
        "  [4] Nettoyage RMT de la matrice de corrélation (Sigma_clean)"
    )

    print(
        "  [5] Analyse du risque systémique "
        "(VaR, ES, CoVaR, MES)"
    )

    print(
        "  [6] Optimisation convexe "
        "(Min-Variance, ERC, Min-Variance bi-pénalisé)"
    )

    print(
        "  [7] Comparaison globale de toutes les méthodes"
    )

    print(
        "\nTous les résultats et figures "
        "ont été calculés et exportés."
    )

    print(
        "\nFichier Excel final :"
    )

    try:
        from utils import OUTPUT_FILE
        print(
            Path(OUTPUT_FILE).resolve()
        )
    except ImportError:
        pass

    print(
        "=" * 70
    )

    return (
        construction_results,
        fx_results,
        rmt_results,
        sr_results,
        optim_results,
        capi_results,
        comparaison_results
    )


# ============================================================
# 5. EXECUTION
# ============================================================

if __name__ == "__main__":
    main()