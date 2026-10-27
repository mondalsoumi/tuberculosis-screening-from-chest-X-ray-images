import argparse, os, numpy as np, pandas as pd, imagehash
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, balanced_accuracy_score

ap = argparse.ArgumentParser(); ap.add_argument("--splits"); ap.add_argument("--root")
a = ap.parse_args()
df = pd.read_csv(a.splits)
files = {}
for d, _, fs in os.walk(a.root):
    for f in fs: files[f] = os.path.join(d, f)
df["real"] = df["path"].apply(lambda p: files[os.path.basename(p.replace("\\", "/"))])

gray, bits = [], []
for p in df["real"]:
    im = Image.open(p).convert("L")
    bits.append(np.array([int(c) for c in bin(int(str(imagehash.phash(im)), 16))[2:].zfill(64)], dtype=np.float32))
    gray.append(np.asarray(im.resize((128, 128), Image.BILINEAR), dtype=np.float32))
gray = np.stack(gray); H = np.stack(bits); y = df["label"].values

# --- near-dup pairs (Hamming<=4) that CROSS splits or classes
ham = 64 - (H @ H.T + (1 - H) @ (1 - H).T)
i, j = np.where(np.triu(ham <= 4, 1))
sp = df["split"].values
rows = []
for a_, b_ in zip(i, j):
    if sp[a_] != sp[b_] or y[a_] != y[b_]:
        x1, x2 = gray[a_].ravel() - gray[a_].mean(), gray[b_].ravel() - gray[b_].mean()
        ncc = float(x1 @ x2 / (np.linalg.norm(x1) * np.linalg.norm(x2) + 1e-9))
        rows.append((df.path[a_], sp[a_], y[a_], df.path[b_], sp[b_], y[b_], int(ham[a_, b_]), ncc))
cp = pd.DataFrame(rows, columns=["A", "splitA", "yA", "B", "splitB", "yB", "ham", "ncc"])
cp.to_csv("outputs/cross_split_pairs.csv", index=False)
print("total pairs ham<=4:", len(i), "| crossing split/class:", len(cp))
if len(cp): print("ncc>=0.95:", (cp.ncc >= .95).sum(), " ncc>=0.98:", (cp.ncc >= .98).sum(), " max ncc: %.3f" % cp.ncc.max())

# --- corrected probes
th = np.stack([np.asarray(Image.fromarray(g).resize((32, 32), Image.BILINEAR)) for g in gray]).reshape(len(gray), -1)
m = np.ones((128, 128), bool); m[13:-13, 13:-13] = False
border = gray[:, m]
f = gray.reshape(len(gray), -1)
stats = np.c_[f.mean(1), f.std(1), np.percentile(f, [1, 5, 25, 50, 75, 95, 99], axis=1).T]
print("majority-class baseline acc: %.3f" % max(y.mean(), 1 - y.mean()))
skf = StratifiedKFold(5, shuffle=True, random_state=42)
for name, X in [("thumb32", th), ("border10%", border), ("intensity", stats)]:
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
    s = cross_val_predict(clf, X, y, cv=skf, method="decision_function")
    print(f"{name:10s} balAcc={balanced_accuracy_score(y, s > 0):.3f}  AUC={roc_auc_score(y, s):.3f}")