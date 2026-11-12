| | Value |
|---|---|
| Model | Qwen/Qwen2.5-0.5B-Instruct (24 layers, hidden size 896) |
| Data | 640 train / 160 test examples (half injected); 10 of 40 injections held out for test |
| Best layer (chosen by train CV) | 12 |
| Delta probe, test ROC AUC / accuracy | 0.977 / 0.931 |
| Plain-activation probe (layer 12), test ROC AUC / accuracy | 0.977 / 0.900 |
| Text-only TF-IDF baseline, test ROC AUC / accuracy | 0.905 / 0.769 |
| Stability: 10 probes, test AUC mean +/- sd | 0.970 +/- 0.008 |
| Stability: share of test inputs where the 10 probes are not unanimous | 0.069 |
| Hard negatives flagged (probe / text baseline) | 0.138 / 0.225 |
