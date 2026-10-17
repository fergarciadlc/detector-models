# Archive

This directory holds exploratory experiments and code that are not used in the paper "How Much AI Is in This Track?". They are kept for reference and may not be runnable as-is. Several of them point at feature or data files that no longer exist, for example `configs/features/stft_db_44k.yaml`.

The paper models live in `configs/experiments/`: `exp_music_cnn_mtg`, `exp_music_cnn_mtg_5s`, `regression_sigmoid_mse_windowed_1s` and `regression_sigmoid_mse_windowed_5s`.

What is archived here, by group:

- Multitask: `aria_multitask_*`, `aria_regression_ai_energy_*` and the `spec_cnn_v1_multitask` model trained a joint binary and regression head on ARIA.
- Fine-tune: the `*_finetune` experiments start from the weights of the binary FMA medium detector, while the paper models are all trained from scratch.
- BCE and ReLU heads: `regression_bce`, `regression_relu` and the `spec_cnn_regression_v2` and `v3` models try a sigmoid with binary cross-entropy and a ReLU output instead of the sigmoid with MSE used in the paper.
- Suno: `exp_foreground_cnn_dzr` and the `fma_medium_suno_3_5` dataset config compare FMA medium against Suno v3.5 tracks from Sonics.
- Mel baselines: `exp_000`, `exp_001`, the `mel_16k_128` features and the `fma_small` dataset are early baselines on 16 kHz mel spectrograms.
- Non-windowed regression: `aria_regression_energy_single_ouput`, `regression_sigmoid_mse_5s` and the `aria` dataset config regress the AI energy ratio per track, while the paper uses the windowed `aria_w1s` and `aria_w5s` datasets.

`src/evaluation/evaluator.py` is an old evaluation loop that nothing imports any more. The live code is in `src/evaluation/evaluators/`.
