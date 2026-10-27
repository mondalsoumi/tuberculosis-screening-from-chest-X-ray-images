import pandas as pd, shutil
shutil.copy("outputs/splits.csv", "outputs/splits_v1_backup.csv")
df = pd.read_csv("outputs/splits.csv")
cp = pd.read_csv("outputs/cross_split_pairs.csv")
cp = cp[cp.ncc >= 0.98]
rank = {"train": 1, "val": 2, "test": 3}
drop = {}
for _, r in cp.iterrows():
    print(r.splitA, r.yA, "|", r.splitB, r.yB, "| ncc %.4f" % r.ncc)
    if r.yA != r.yB:                       # same pixels, conflicting labels: drop both
        drop[r.A] = drop[r.B] = "label conflict"
    else:
        d = r.A if rank[r.splitA] < rank[r.splitB] else r.B
        drop[d] = "copy of " + (r.B if d == r.A else r.A)
out = df[~df.path.isin(drop)]
out.to_csv("outputs/splits.csv", index=False)
pd.Series(drop, name="reason").rename_axis("path").to_csv("outputs/dropped_leaks.csv")
print("dropped:", len(drop))
print(out.groupby(["label", "split"]).size().unstack())