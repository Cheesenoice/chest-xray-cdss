# Multi-Backbone Benchmark Results (Mean ± Std over 3 Seeds)

- **Tested Backbones:** resnet18, densenet121, efficientnet_b0
- **Random Seeds:** [42, 7, 123]
- **Training Epochs per Run:** 15

## 1. Internal Test Set Performance

| Backbone | Accuracy | Precision (Macro) | Recall (Macro) | F1 Score (Macro) | AUC (Macro) |
|---|---|---|---|---|---|
| **resnet18** | 0.8498 ± 0.0033 | 0.8358 ± 0.0041 | 0.8489 ± 0.0054 | **0.8409 ± 0.0022** | 0.9538 ± 0.0020 |
| **densenet121** | 0.8455 ± 0.0022 | 0.8349 ± 0.0115 | 0.8328 ± 0.0086 | **0.8322 ± 0.0055** | 0.9512 ± 0.0024 |
| **efficientnet_b0** | 0.8383 ± 0.0091 | 0.8306 ± 0.0108 | 0.8248 ± 0.0070 | **0.8269 ± 0.0088** | 0.9493 ± 0.0015 |

## 2. External Test Set Performance (Held-Out Montgomery Source)

| Backbone | Accuracy | Precision (Macro) | Recall (Macro) | F1 Score (Macro) | AUC (Macro) |
|---|---|---|---|---|---|
| **resnet18** | 0.6932 ± 0.0304 | 0.3798 ± 0.0126 | 0.3361 ± 0.0119 | **0.3535 ± 0.0100** | 0.7606 ± 0.0072 |
| **densenet121** | 0.6860 ± 0.0723 | 0.4083 ± 0.0175 | 0.3310 ± 0.0373 | **0.3575 ± 0.0293** | 0.8296 ± 0.0613 |
| **efficientnet_b0** | 0.5411 ± 0.0616 | 0.4245 ± 0.0092 | 0.2495 ± 0.0284 | **0.2836 ± 0.0200** | 0.7208 ± 0.0360 |
