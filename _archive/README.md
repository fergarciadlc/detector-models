# Archive

This directory holds exploratory experiments and code that are not used in the paper "How Much AI Is in This Track?". They are kept for reference only. They may not run as-is, since the code around them has moved on and some of them need data that is not part of the release.

The paper models live in `configs/experiments/`: `exp_music_cnn_mtg`, `exp_music_cnn_mtg_5s`, `regression_sigmoid_mse_windowed_1s` and `regression_sigmoid_mse_windowed_5s`.

What is archived here:

- Multitask: `aria_multitask_*` and the `spec_cnn_v1_multitask` model trained a joint binary and regression head on ARIA.
- Fine-tune: the `*_finetune` experiments start from pretrained weights, while the paper models are all trained from scratch.
- BCE and ReLU heads: `regression_bce`, `regression_relu` and the `spec_cnn_regression_v2` and `v3` models compare other output heads against the sigmoid plus MSE regressor used in the paper.
- Suno: the `fma_medium_suno_3_5` dataset config was an exploration of Suno 3.5 generated tracks as the fake class.
- Mel baselines: `mel_16k_128` and the early `exp_000`, `exp_001` and `exp_foreground_cnn_dzr` experiments used mel spectrogram or other early setups before the STFT dB pipeline.
- Non-windowed regression: `aria_regression_*`, `regression_sigmoid_mse_5s` and the `aria` dataset config regress the AI energy ratio on whole tracks, while the paper uses the windowed `aria_w1s` and `aria_w5s` datasets.

`src/evaluation/evaluator.py` is an old evaluation loop that nothing imports any more. The live code is in `src/evaluation/evaluators/`.
