import numpy as np, sys
C = "data/cache_full"
apply = "--apply" in sys.argv
def load(s): return np.load(f"{C}/{s}_X.npy"), np.load(f"{C}/{s}_y.npy")
def feat(X):
    f = X.reshape(len(X), 56, 4, 56, 4).mean((2, 4)).reshape(len(X), -1).astype(np.float32)
    f -= f.mean(1, keepdims=True)
    f /= np.linalg.norm(f, axis=1, keepdims=True) + 1e-9
    return f
Xt, yt = load("train"); Xv, yv = load("val"); Xs, ys = load("test")
for n, X, y in [("train", Xt, yt), ("val", Xv, yv), ("test", Xs, ys)]:
    print(n, X.shape, X.dtype, "TB:", int(y.sum()), "Normal:", int((y == 0).sum()), "mean px: %.1f" % X[:200].mean())
assert (len(Xv), len(Xs)) == (832, 335) and int(yv.sum()) == 134 and int(ys.sum()) == 64, "WRONG CACHE - STOP"
Ft, Fv, Fs = feat(Xt), feat(Xv), feat(Xs)
bad = set()
for name, Fa, ya, Fb, yb in [("val->train", Fv, yv, Ft, yt), ("test->train", Fs, ys, Ft, yt), ("test->val", Fs, ys, Fv, yv)]:
    S = Fa @ Fb.T; m = S.max(1); j = S.argmax(1)
    print(name, ">=.98:", int((m >= .98).sum()), ">=.95:", int((m >= .95).sum()), ">=.90:", int((m >= .90).sum()),
          "| label mismatch @.98:", int(((m >= .98) & (ya != yb[j])).sum()))
    if name.endswith("train"): bad |= set(j[m >= .98].tolist())
print("train rows that twin a val/test image:", len(bad))
if apply and bad:
    keep = np.array([i for i in range(len(Xt)) if i not in bad])
    np.save(f"{C}/train_X.npy", Xt[keep]); np.save(f"{C}/train_y.npy", yt[keep])
    print("new train:", len(keep), "TB:", int(yt[keep].sum()))