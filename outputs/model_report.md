# Model Report
> GMACs counted with [thop](https://github.com/Lyken17/pytorch-OpCounter) (MACs = multiply-accumulate ops; GMACs = 10⁹ MACs).

| Model | Params | FP32 MB | GMACs @ 224 | ECA params |
|---|---|---|---|---|
| LiteTBNet (ECA=True) | 262,700 | 1.05 | 0.134 | 35 |
| LiteTBNet (ECA=False) | 262,665 | 1.05 | 0.134 | 0 |
| Swin-Tiny | 27,520,123 | 110.08 | 4.371 | – |
| InceptionV3 | 21,787,617 | 87.15 | 2.836 | – |
