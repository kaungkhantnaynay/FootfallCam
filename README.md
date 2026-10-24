# FootfallCam staff identification prototype

A local Python pipeline that detects people in video, checks for a supplied visual tag, tracks short-lived identities, and exports frame-level staff predictions and coordinates.

This is an assessment prototype. The supplied development clip does not establish accuracy on new videos. A prediction of `staff_present=0` means the pipeline did not confirm staff in that frame; it does not prove nobody was present.

## Set up

Use Python 3.12 and `uv` from the project root:

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.lock.txt
uv pip install --python .venv/bin/python --no-deps -e .
.venv/bin/python scripts/download_model.py
```

The download script fetches Ultralytics YOLO11n weights and checks their SHA-256 hash. The tag reference crops and locally calibrated verifier are included in the repository. Bring your own video; input videos and generated results are excluded from Git.

## Run

```bash
.venv/bin/python main.py --input /path/to/video.mp4 --output results/my-run
```

Choose a new output directory for each run. Existing results are not overwritten. The default runs on CPU and may take several minutes. Use `.venv/bin/python main.py --help` for other options, including `--mode causal` and `--no-video`.

| Output | Meaning |
| --- | --- |
| `frames.csv` | One row per decoded frame, including frames without confirmed staff |
| `staff_observations.csv` | Confirmed staff track and original-frame coordinates |
| `observations.jsonl` | All person observations, including tentative and rejected evidence |
| `annotated.mp4` | Video with staff labels, track IDs, coordinates, and tag boxes |
| `resolved_config.json` | Settings used for the run |
| `run_manifest.json` | Input and model hashes, versions, timing, and run metadata |
| `track_diagnostics.json` | Tag-hit and movement summaries per track |

To produce a readable summary and plot after a run:

```bash
.venv/bin/python scripts/summarise_run.py --run results/my-run
```

## How it works

`main.py` starts `src/footfallcam/pipeline.py`. A pretrained YOLO11n model proposes person boxes in four orientations. Reference matching and a small SVM verifier check possible tag patches inside those boxes. A custom tracker associates detections across frames. The final decision requires repeated tag evidence and movement. The pipeline then writes the files above and, unless disabled, renders the annotated video.

The person detector was not trained or fine-tuned in this project. The tag verifier was calibrated from a small set of reference and development-video patches; those examples are not an independent test set. Development used AI coding assistance.

## Checks and attribution

```bash
.venv/bin/python -m pytest -q
```

These tests cover coordinate mapping, tracking, temporal decisions, evaluation rules, and output contracts. Passing them does not measure accuracy on unseen video.

The pretrained person model is [Ultralytics YOLO11n](https://docs.ultralytics.com/models/yolo11/); its weights come from the [Ultralytics release asset](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt). Review the [Ultralytics licensing terms](https://www.ultralytics.com/license) before redistribution or commercial use. The tag reference images are derived from assessment material and have separate provenance in `data/reference/reference_manifest.json`.
