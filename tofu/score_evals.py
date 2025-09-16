import os
import json
import math
from collections import defaultdict

methods = ["tempsave", "SimNPO", "UNDIAL", "IdkDPO", "RMU", "IdkNLL", "GradDiff", "AltPO", "NPO", "DAWI", "RETAIN", "OVERTRAINED"]

RETRAIN_STATS = {
    "exact_memorization": 0.5860680475085974,
    "extraction_strength": 0.05975501511819896,
    "forget_Q_A_PARA_Prob": 0.05349626573151909,
    "forget_Q_A_PARA_ROUGE": 0.2811751970102694,
    "forget_Q_A_PERT_Prob": 0.0406794768360196,
    "forget_Q_A_PERT_ROUGE": 0.2716223515131106,
    "forget_Q_A_Prob": 0.1160702278232202,
    "forget_Q_A_ROUGE": 0.37943121819911346,
    "forget_Q_A_gibberish": 0.8981131750717759,
    "forget_quality": 0.9999999999893432,
    "forget_truth_ratio": 0.6270187030823124,
    "mia_gradnorm": 0.34169062499999997,
    "mia_loss": 0.3869125,
    "mia_min_k": 0.38209375,
    "mia_min_k_plus_plus": 0.4758875,
    "mia_zlib": 0.30844375,
    "model_utility": 0.5892923390686856,
    "privleak": 0.08402340506019135,
    "ra_Q_A_PERT_Prob": 0.006271589553077016,
    "ra_Q_A_Prob": 0.013936679118596657,
    "ra_Q_A_Prob_normalised": 0.3915799226316357,
    "ra_Q_A_ROUGE": 0.8145,
    "ra_Truth_Ratio": 0.5022027051450156,
    "retain_Q_A_PARA_Prob": 0.08653909503598697,
    "retain_Q_A_PERT_Prob": 0.0378325767547416,
    "retain_Q_A_Prob": 0.8802840007841587,
    "retain_Q_A_ROUGE": 0.8264372396779979,
    "retain_Truth_Ratio": 0.5138097597002513,
    "wf_Q_A_PERT_Prob": 0.001910109234122783,
    "wf_Q_A_Prob": 0.004878307801588089,
    "wf_Q_A_Prob_normalised": 0.4279515925645054,
    "wf_Q_A_ROUGE": 0.8022792022792024,
    "wf_Truth_Ratio": 0.6156096615954564
}

def clamp01(x):
    if x is None:
        return None
    return max(0.0, min(1.0, float(x)))

def inv01(x):
    """Return 1 - x (clamped to [0,1]) if x is not None, else None."""
    if x is None:
        return None
    return clamp01(1.0 - float(x))

def hmean(values, eps=1e-12):
    """Harmonic mean on [0,1] metrics; ignores None; returns None if no valid values."""
    cleaned = [clamp01(v) for v in values if v is not None]
    if not cleaned:
        return None
    if any(v <= eps for v in cleaned):
        return 0.0
    return len(cleaned) / sum(1.0 / v for v in cleaned)

def compute_scores(stats):
    """
    Implements OpenUnlearning metric aggregation:
      Memorization = HM(1-ES, 1-EM, 1-ParaProb, 1-TruthRatio)
      Utility      = HM(MU, Fluency), with Fluency = 1 - gibberish (forget prompts)
      Privacy      = HM(sLOSS, sZLib, sMinK, sMinKpp)  (assumed already scaled to [0,1])
      Tuning Obj   = HM(Memorization, Utility)
      Overall      = HM(Memorization, Utility, Privacy)  (if Privacy available, else = Tuning)
    """

    ES   = stats.get("extraction_strength")
    EM   = stats.get("exact_memorization")
    PARA = stats.get("forget_Q_A_PARA_Prob")
    TR   = stats.get("forget_truth_ratio")
    
    retrain_ES   = RETRAIN_STATS.get("extraction_strength")
    retrain_EM   = RETRAIN_STATS.get("exact_memorization")
    retrain_PARA = RETRAIN_STATS.get("forget_Q_A_PARA_Prob")
    retrain_TR   = RETRAIN_STATS.get("forget_truth_ratio") 

    mem_components = [inv01(ES), inv01(EM), 
                      inv01(PARA), inv01(TR)]
    mem_score = hmean(mem_components)

    MU = stats.get("model_utility")
    retrain_MU = RETRAIN_STATS.get("model_utility") 
    MU = min(MU / retrain_MU, 1)             
    gib = stats.get("forget_Q_A_gibberish")
    fluency = None if gib is None else clamp01(1.0 - float(gib))
    retrain_gib = RETRAIN_STATS.get("forget_Q_A_gibberish")
    retain_fluency = None if gib is None else clamp01(1.0 - float(retrain_gib))
    fluency = min(fluency / retain_fluency, 1)
    if fluency is None:
        utility_score = clamp01(MU) if MU is not None else None
    else:
        utility_score = hmean([MU, ])

    def auc_convert(x):
        return 1 - 2 * abs(x - 0.5)
    
    sLOSS   = auc_convert(stats.get("mia_loss"))
    sZLib   = auc_convert(stats.get("mia_zlib"))
    sMinK   = auc_convert(stats.get("mia_min_k"))
    sMinKpp = auc_convert(stats.get("mia_min_k_plus_plus"))
    
    retrain_sLOSS   = auc_convert(RETRAIN_STATS.get("mia_loss"))
    retrain_sZLib   = auc_convert(RETRAIN_STATS.get("mia_zlib"))
    retrain_sMinK   = auc_convert(RETRAIN_STATS.get("mia_min_k"))
    retrain_sMinKpp = auc_convert(RETRAIN_STATS.get("mia_min_k_plus_plus"))
    
    
    mia_vals = [min(sLOSS / retrain_sLOSS, 1), min(sZLib / retrain_sZLib, 1), min(sMinK / retrain_sMinK, 1), min(sMinKpp / retrain_sMinKpp, 1)]
    if any(v is not None for v in mia_vals):
        privacy_score = hmean(mia_vals)
    else:
        privacy_score = None

    tuning_obj = hmean([mem_score, utility_score])
    overall = hmean([mem_score, utility_score, privacy_score]) if privacy_score is not None else tuning_obj

    return {
        "memorization": mem_score,
        "utility": utility_score,
        "privacy": privacy_score,
        "tuning_objective": tuning_obj,
        "overall": overall,
        "privleak": stats.get("privleak"),
        "forget_quality": stats.get("forget_quality")
    }

import argparse
parser = argparse.ArgumentParser(description="Example of --test flag.")
parser.add_argument("--path", type=str, required=True, help="Path to folder containing evals")
args = parser.parse_args()
path = args.path
print("PATH: ", path)

results = [] 
by_method = defaultdict(list)

folders = [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]
for save_point in folders:
    method_type = None
    for method_name in methods:
        if method_name in save_point:
            method_type = method_name
            break

    if method_type is None:
        raise ValueError(f"{save_point} not categorized")

    summary_path = os.path.join(path, save_point, "TOFU_SUMMARY.json")
    if not os.path.exists(summary_path):
        continue

    try:
        with open(summary_path, "r") as f:
            statsx = json.load(f)
    except Exception as e:
        print(f"Failed to read {summary_path}: {e}")
        continue

    scores = compute_scores(statsx)
    record = {
        "method": method_type,
        "save_point": save_point,
        **scores,
        "model_utility_raw": statsx.get("model_utility"),
        "gibberish_raw": statsx.get("forget_Q_A_gibberish"),
        "extraction_strength": statsx.get("extraction_strength"),
        "exact_memorization": statsx.get("exact_memorization"),
        "forget_truth_ratio": statsx.get("forget_truth_ratio"),
        "forget_Q_A_PARA_Prob": statsx.get("forget_Q_A_PARA_Prob"),
        "mia_loss": statsx.get("mia_loss"),
        "mia_zlib": statsx.get("mia_zlib"),
        "mia_min_k": statsx.get("mia_min_k"),
        "mia_min_k_plus_plus": statsx.get("mia_min_k_plus_plus"),
    }
    results.append(record)
    by_method[method_type].append(record)

def sort_and_print(title, rows, key="overall", top_k=5):
    rows_sorted = sorted(
        [r for r in rows if r.get(key) is not None],
        key=lambda r: r[key],
        reverse=True
    )
    print(f"\n=== {title} (sorted by {key} desc) ===")
    for i, r in enumerate(rows_sorted[:top_k], 1):
        print(f"Privleak: {r['privleak']}, Forget Quality: {r['forget_quality']}")
        print(
            f"{i:2d}. {r['method']:8s} | {r['save_point']:40s} "
            f"| overall={r['overall']:.4f} | tune={r['tuning_objective']:.4f} "
            f"| mem={r['memorization']:.4f} | util={r['utility']:.4f} "
            f"| priv={r['privacy']:.4f}" if r['privacy'] is not None else
            f"{i:2d}. {r['method']:8s} | {r['save_point']:40s} "
            f"| overall={r['overall']:.4f} | tune={r['tuning_objective']:.4f} "
            f"| mem={r['memorization']:.4f} | util={r['utility']:.4f} | priv=NA" 
        )

for m, rows in by_method.items():
    sort_and_print(f"TOP for {m}", rows, key="tuning_objective", top_k=1)


