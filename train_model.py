"""Trains and evaluates the crowd-level classifiers; saves the best model (Random Forest) and metrics."""
import json, joblib
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support, confusion_matrix,
                             classification_report)
from features import encode, FEATURES, LEVELS

df = pd.read_csv("data/crowd_data.csv", keep_default_na=False)
X, y = encode(df), df["crowd_level"].map({l: i for i, l in enumerate(LEVELS)})
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

models = {
    "Random Forest": RandomForestClassifier(n_estimators=200, min_samples_leaf=3, max_features=0.7,
                                            random_state=42, n_jobs=-1),
    "Decision Tree": DecisionTreeClassifier(max_depth=12, random_state=42),
    "KNN (k=7)": make_pipeline(StandardScaler(), KNeighborsClassifier(7)),
    "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
}
results = {}
for name, m in models.items():
    m.fit(Xtr, ytr); p = m.predict(Xte)
    pr, rc, f1, _ = precision_recall_fscore_support(yte, p, average="weighted", zero_division=0)
    results[name] = dict(accuracy=accuracy_score(yte, p), precision=pr, recall=rc, f1=f1)
    print(name, {k: round(v, 4) for k, v in results[name].items()})

rf = models["Random Forest"]; pred = rf.predict(Xte)
cv = cross_val_score(RandomForestClassifier(n_estimators=100, min_samples_leaf=3, max_features=0.7,
                     random_state=42, n_jobs=-1), X, y,
                     cv=StratifiedKFold(5, shuffle=True, random_state=42), scoring="accuracy")
cm = confusion_matrix(yte, pred)
rep = classification_report(yte, pred, target_names=LEVELS, output_dict=True, zero_division=0)
imp = sorted(zip(FEATURES, rf.feature_importances_), key=lambda t: -t[1])
metrics = dict(rows=len(df), train=len(Xtr), test=len(Xte), models=results, cv_mean=cv.mean(), cv_std=cv.std(),
               confusion=cm.tolist(), report=rep, importance=[(f, float(v)) for f, v in imp],
               class_share=df.crowd_level.value_counts(normalize=True).to_dict())
json.dump(metrics, open("model/metrics.json", "w"), indent=2)
joblib.dump(rf, "model/crowd_rf.joblib", compress=3)
print("CV", cv.mean().round(4), cv.std().round(4)); print(cm); print(imp[:8])

# ---- figures ----
P = ["#4B2E83", "#F2A900", "#C0392B", "#2E86AB"]
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(figsize=(5, 3.4))
s = df.crowd_level.value_counts().reindex(LEVELS)
ax.bar(LEVELS, s.values, color=["#6AA84F", "#F2A900", "#E67E22", "#C0392B"])
for i, v in enumerate(s.values): ax.text(i, v, f"{v/len(df):.1%}", ha="center", va="bottom")
ax.set_ylabel("Records"); ax.set_title("Class distribution"); fig.tight_layout(); fig.savefig("docs/fig_class_dist.png", dpi=200); plt.close()

fig, ax = plt.subplots(figsize=(4.6, 4))
ax.imshow(cm, cmap="Purples")
for i in range(4):
    for j in range(4): ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max()/2 else "black")
ax.set_xticks(range(4)); ax.set_yticks(range(4)); ax.set_xticklabels(LEVELS, rotation=30); ax.set_yticklabels(LEVELS)
ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title("Confusion matrix (Random Forest)")
fig.tight_layout(); fig.savefig("docs/fig_confusion.png", dpi=200); plt.close()

top = imp[:10][::-1]
fig, ax = plt.subplots(figsize=(5.4, 3.6))
ax.barh([t[0] for t in top], [t[1] for t in top], color=P[0]); ax.set_title("Top 10 feature importances")
fig.tight_layout(); fig.savefig("docs/fig_importance.png", dpi=200); plt.close()

fig, ax = plt.subplots(figsize=(5.4, 3.4))
names = list(results); acc = [results[n]["accuracy"] * 100 for n in names]
ax.bar(names, acc, color=[P[0], P[1], P[3], P[2]]); ax.set_ylim(min(acc) - 10, 100)
for i, v in enumerate(acc): ax.text(i, v + .3, f"{v:.1f}%", ha="center")
ax.set_ylabel("Test accuracy (%)"); ax.set_title("Model comparison"); plt.xticks(rotation=12)
fig.tight_layout(); fig.savefig("docs/fig_models.png", dpi=200); plt.close()

df["date"] = pd.to_datetime(df["date"]); df["m"] = df.date.dt.month
mm = df.groupby("m").crowd_ratio.mean()
fig, ax = plt.subplots(figsize=(5.4, 3.2))
ax.plot(mm.index, mm.values, marker="o", color=P[0]); ax.set_xticks(range(1, 13))
ax.set_xlabel("Month"); ax.set_ylabel("Mean crowd ratio"); ax.set_title("Average crowd by month")
fig.tight_layout(); fig.savefig("docs/fig_monthly.png", dpi=200); plt.close()
