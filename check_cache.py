import numpy as np
for s in ["train", "val", "test"]:
    X = np.load(f"data/cache/{s}_X.npy", mmap_mode="r"); y = np.load(f"data/cache/{s}_y.npy")
    print(s, X.shape, X.dtype, "TB:", int(y.sum()), "Normal:", int((y == 0).sum()), "mean px: %.1f" % X[:200].mean())