"""
research-notes 16~18 노트용 예제 그림 생성 스크립트 (재현 가능).
- 데이터: sklearn load_breast_cancer (악성=1 양성=0), 고정 시드.
- 생성물: ROC+DeLong, calibration(reliability), SHAP beeswarm, SHAP dependence.
실행:  python3 assets/make_figures.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.calibration import calibration_curve
import xgboost as xgb
import shap

SEED = 42
np.random.seed(SEED)
OUT = os.path.dirname(os.path.abspath(__file__))


# ----------------------------------------------------------------------
# Fast DeLong (Sun & Xu, 2014) — correlated ROC AUC 비교의 분산/공분산.
# 구현 출처 개념: E.R. DeLong et al. (1988); 고속화 X. Sun & W. Xu (2014).
# ----------------------------------------------------------------------
def _compute_midrank(x):
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T
    return T2


def delong_roc_variance(y_true, y_score):
    """단일 모델 AUC와 그 분산(추가 검증용)."""
    order = np.argsort(-y_score)
    label_1 = y_true[order] == 1
    m = int(label_1.sum())
    n = len(y_true) - m
    pos = y_score[order][label_1]
    neg = y_score[order][~label_1]
    return _delong(pos, neg)


def _delong(pos, neg):
    m, n = len(pos), len(neg)
    tx = _compute_midrank(pos)
    ty = _compute_midrank(neg)
    txy = _compute_midrank(np.concatenate([pos, neg]))
    auc = (txy[:m].sum() - m * (m + 1) / 2) / (m * n)
    v01 = (txy[:m] - tx) / n
    v10 = 1 - (txy[m:] - ty) / m
    s = v01.var(ddof=1) / m + v10.var(ddof=1) / n
    return auc, s, v01, v10


def delong_test_paired(y_true, score_a, score_b):
    """동일 표본 두 모델 AUC 비교 → (auc_a, auc_b, p_value)."""
    from scipy import stats
    pos_mask = y_true == 1
    a_pos, a_neg = score_a[pos_mask], score_a[~pos_mask]
    b_pos, b_neg = score_b[pos_mask], score_b[~pos_mask]
    m, n = pos_mask.sum(), (~pos_mask).sum()
    auc_a, _, va01, va10 = _delong(a_pos, a_neg)
    auc_b, _, vb01, vb10 = _delong(b_pos, b_neg)
    # 공분산 (상관 반영)
    cov01 = np.cov(np.vstack([va01, vb01]))
    cov10 = np.cov(np.vstack([va10, vb10]))
    var = (cov01[0, 0] / m + cov10[0, 0] / n,
           cov01[1, 1] / m + cov10[1, 1] / n,
           cov01[0, 1] / m + cov10[0, 1] / n)
    se = np.sqrt(var[0] + var[1] - 2 * var[2])
    z = (auc_a - auc_b) / se if se > 0 else 0.0
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return auc_a, auc_b, p


# ----------------------------------------------------------------------
# 데이터 + 두 모델 (선형 vs 부스팅)
# ----------------------------------------------------------------------
data = load_breast_cancer()
X, y = data.data, data.target
# sklearn 규약: target 0=malignant,1=benign → "악성=양성클래스"로 뒤집어 임상감각에 맞춤
y = 1 - y
feat = list(data.feature_names)
# 교육용으로 난이도를 의료현실 수준(AUC~0.85-0.92)으로: feature 잡음 + 라벨 잡음 주입
rng = np.random.RandomState(SEED)
X = X + rng.normal(0, X.std(0) * 1.8, size=X.shape)
flip = rng.rand(len(y)) < 0.08
y = np.where(flip, 1 - y, y)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=SEED, stratify=y)

logit = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, C=0.05))
logit.fit(Xtr, ytr)
p_logit = logit.predict_proba(Xte)[:, 1]

booster = xgb.XGBClassifier(
    n_estimators=250, max_depth=4, learning_rate=0.15,
    subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
    eval_metric="logloss", random_state=SEED,
)
booster.fit(Xtr, ytr)
p_boost = booster.predict_proba(Xte)[:, 1]

auc_a, auc_b, p_delong = delong_test_paired(yte.astype(int), p_logit, p_boost)
print(f"AUC logistic={auc_a:.3f}  AUC xgboost={auc_b:.3f}  DeLong p={p_delong:.3f}")


# ----------------------------------------------------------------------
# Fig 1. ROC + DeLong
# ----------------------------------------------------------------------
plt.figure(figsize=(5.2, 5.0))
for name, p, c in [("Logistic (L2)", p_logit, "#1f77b4"), ("XGBoost", p_boost, "#d62728")]:
    fpr, tpr, _ = roc_curve(yte, p)
    plt.plot(fpr, tpr, color=c, lw=2, label=f"{name}: AUC={roc_auc_score(yte,p):.3f}")
plt.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.6)
plt.xlabel("1 − Specificity (FPR)"); plt.ylabel("Sensitivity (TPR)")
plt.title("ROC curves — two models on one test set")
plt.legend(loc="lower right", fontsize=9)
plt.text(0.42, 0.10, f"DeLong paired test\np = {p_delong:.3f}",
         fontsize=9, bbox=dict(boxstyle="round", fc="#fff7e6", ec="#e0a800"))
plt.tight_layout(); plt.savefig(f"{OUT}/fig_roc_delong.png", dpi=150); plt.close()


# ----------------------------------------------------------------------
# Fig 2. Calibration (reliability diagram) — 보정 우수 vs 과신
# ----------------------------------------------------------------------
plt.figure(figsize=(5.2, 5.0))
for name, p, c in [("Logistic (well-calibrated)", p_logit, "#1f77b4"),
                   ("XGBoost (sharper)", p_boost, "#d62728")]:
    frac_pos, mean_pred = calibration_curve(yte, p, n_bins=8, strategy="quantile")
    plt.plot(mean_pred, frac_pos, "o-", color=c, lw=2, label=name)
plt.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.6, label="Perfect calibration")
plt.xlabel("Mean predicted probability"); plt.ylabel("Observed frequency")
plt.title("Calibration plot (reliability diagram)")
plt.legend(loc="upper left", fontsize=8.5)
plt.tight_layout(); plt.savefig(f"{OUT}/fig_calibration.png", dpi=150); plt.close()


# ----------------------------------------------------------------------
# Fig 3-4. SHAP (TreeExplainer, interventional) — beeswarm & dependence
# ----------------------------------------------------------------------
explainer = shap.TreeExplainer(booster, feature_perturbation="interventional", data=Xtr[:200])
sv = explainer(Xte, check_additivity=False)
sv.feature_names = feat

plt.figure()
shap.summary_plot(sv.values, Xte, feature_names=feat, max_display=12, show=False)
plt.title("SHAP summary (beeswarm) — XGBoost", fontsize=11)
plt.tight_layout(); plt.savefig(f"{OUT}/fig_shap_beeswarm.png", dpi=150, bbox_inches="tight"); plt.close()

top = feat[int(np.argmax(np.abs(sv.values).mean(0)))]
plt.figure()
shap.dependence_plot(top, sv.values, Xte, feature_names=feat, show=False)
plt.title(f"SHAP dependence — {top}", fontsize=11)
plt.tight_layout(); plt.savefig(f"{OUT}/fig_shap_dependence.png", dpi=150, bbox_inches="tight"); plt.close()

print("Saved figures to", OUT)
print("top feature:", top)
