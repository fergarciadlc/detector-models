# Reproducing the paper

How to run the experiments of the ISMIR 2026 paper [How Much AI Is in This Track?](https://arxiv.org/abs/2608.07285) with this repository. The results are reported in the paper (Table 2).

The binary detector reproduces the CNN of Afchar et al. (2025) and is used as a probe: its fake probability is read as an estimate of α. The regression CNN has the same backbone with a single sigmoid output, trained with mean squared error on the per-window α. The binary models use FMA-Medium and its Encodec 3 kbps reconstruction. The regression models use ARIA, built from MoisesDB with Encodec 3 kbps.

| Paper model | Experiment config | Dataset config | Window | Feature config |
|---|---|---|---|---|
| Binary CNN, 1 s | `configs/experiments/exp_music_cnn_mtg.yaml` | `fma_medium` | 1 s | `stft_db_44k_full` |
| Binary CNN, 5 s | `configs/experiments/exp_music_cnn_mtg_5s.yaml` | `fma_medium` | 5 s | `stft_db_44k_8s` (sliced, see step 2) |
| Regression CNN, 1 s | `configs/experiments/regression_sigmoid_mse_windowed_1s.yaml` | `aria_w1s` | 1 s | `stft_db_44k_60s` |
| Regression CNN, 5 s | `configs/experiments/regression_sigmoid_mse_windowed_5s.yaml` | `aria_w5s` | 5 s | `stft_db_44k_60s` |

The output folders use the feature `name`: `stft_db_44k_full.yaml` has name `stft_db_44k`.

## 1. Build the data

ARIA stems and mixes come from MoisesDB (see the [aria README](https://github.com/fergarciadlc/aria)). Compute the per-window α and the window manifests with the aria scripts:

```bash
# in the aria repository
python scripts/compute_metrics.py --aria-path $ARIA_ROOT --window 5 --output-dir $ARIA_ROOT/metrics/5s
python scripts/create_splits.py --aria-path $ARIA_ROOT \
    --metrics-csv $ARIA_ROOT/metrics/5s/mix_metrics.csv \
    --split-from $ARIA_ROOT/splits/canonical_v2.jsonl \
    --output <this repo>/data/datasets/aria_w5s/canonical_v3.jsonl
```

Repeat with `--window 1` and output `<this repo>/data/datasets/aria_w1s/canonical_v4.jsonl` for the 1 s models. These are the `predefined_path` values of `configs/datasets/aria_w5s.yaml` and `configs/datasets/aria_w1s.yaml`. The paper split must be reused with `--split-from canonical_v2.jsonl`, since it cannot be regenerated from the audio (the aria README explains why).

Place the FMA data at the `paths` of `configs/datasets/fma_medium.yaml`: FMA-Medium (real) and its Encodec 3 kbps rebuild (fake). `DATASETS_DIR` then holds `FMA/fma_medium`, `FMA/fma_medium_rebuilt_mp3_44100_256` and `aria`.

## 2. Extract features

```bash
make prepare features EXP=configs/experiments/regression_sigmoid_mse_windowed_5s.yaml WORKERS=8
make prepare features EXP=configs/experiments/regression_sigmoid_mse_windowed_1s.yaml WORKERS=8
make prepare features EXP=configs/experiments/exp_music_cnn_mtg.yaml WORKERS=8
```

The 5 s binary model needs the third command first, because its features are slices of the 1 s binary features. Slicing reads `data/features/stft_db_44k/fma_medium/feature_manifest.json`:

```bash
python scripts/slice_features_duration.py \
    --src-manifest data/features/stft_db_44k/fma_medium/feature_manifest.json \
    --out-dir data/features/stft_db_44k_8s/fma_medium \
    --feature-name stft_db_44k_8s \
    --keep-sec 8 --min-crop-frames 427 --workers 8
```

## 3. Train and evaluate

```bash
make train-eval EXP=configs/experiments/regression_sigmoid_mse_windowed_5s.yaml
make train-eval EXP=configs/experiments/exp_music_cnn_mtg_5s.yaml
```

The same command applies to the other two configs. `make train` and `make evaluate` run the stages separately, and `MANIFEST=<path>` overrides the feature manifest. Checkpoints go to `data/artifacts/{experiment name}/best_model.keras`. The windowed experiments write `eval/windows/` and `eval/tracks/`, and the track-level regression metrics reported in the paper are in `eval/tracks/eval_metrics.csv`. The binary experiments write their metrics to `eval/`.

## 4. Binary α-estimate rows

The paper's "α-est." rows apply a binary model to ARIA mixes and compare its fake probability with α. Run these commands in the container (see Quick start in the [README](README.md#quick-start)) or in a local environment where `/datasets/aria` exists. Copy `canonical_v2.jsonl` from the aria dataset to `data/datasets/aria/splits/` first.

```bash
python - <<'EOF' > data/mixes_test.txt
import json
for line in open("data/datasets/aria/splits/canonical_v2.jsonl"):
    row = json.loads(line)
    if row.get("split") == "test":
        print(f"/datasets/aria/{row['mix_path']}")
EOF

python -m pipelines.predict \
    --model-path data/artifacts/cnn_mtg_stft_db_44k_fma_medium_encodec_5s/best_model.keras \
    --input data/mixes_test.txt --workers 8 --output-dir data/predictions/binary_5s

python scripts/binary_regression_metrics.py \
    --predictions data/predictions/binary_5s/predictions.csv \
    --experiment cnn_mtg_stft_db_44k_fma_medium_encodec_5s \
    --aria-splits data/datasets/aria/splits/canonical_v2.jsonl --split test
```

For the 1 s row use the experiment `cnn_mtg_stft_db_44k_fma_medium_encodec`. The metrics script writes `joined_predictions.csv` and `regression_metrics_vs_aria_targets.csv` in `analysis/` next to the predictions.

## Notes

- **8 s binary features.** `stft_db_44k_8s` comes from slicing the first 8 s of the full-length `stft_db_44k` features. Re-extracting with `max_duration_sec: 8` selects a different start offset and takes the dB reference over the shorter segment, so it gives different features.
- **Evaluation span.** The binary "α-est." rows are scored on whole tracks: `pipelines.predict` averages window scores over the full mix and `binary_regression_metrics.py` compares them with the full-mix α of `canonical_v2`. The regression rows come from `pipelines.evaluate`, which averages window predictions per mix over the 60 s excerpt covered by the `stft_db_44k_60s` features and compares them with the per-window α. The two kinds of row cover different spans of each mix.
- **Deviations from Afchar et al.**
  - The spectrogram is in dB (`librosa.amplitude_to_db`, `ref=np.max`), and the normalisation constants (mean -4, std 3) come from the log10-power setup of the original code.
  - The frame count formula assumes `center=False` and `librosa.stft` uses `center=True`, so windows start about 23 ms early.
  - The binary FMA split is made per file.
  - Silent windows have α = 0, a convention inherited from aria.
- **Environment.** The paper models were trained with `docker/Dockerfile.gpu` (TensorFlow 2.11). `pyproject.toml` pins TensorFlow 2.15 for local use.
