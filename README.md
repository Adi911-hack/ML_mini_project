# Colatron — Machine Learning Mini Project

This repository explores the Stanford Colatron text-to-speech work using an **adapted Support Vector Regression (SVR) demonstration** and a duration-calibration experiment.

**Important:** The original source release has missing training resources and pretrained weights. The adapted SVR demo is not a faithful reproduction of the original Seq2Seq system. No improvement in speech intelligibility has been established.

## Core Python code

- `colatron_custom_eval/work/colatron_reproduction/run_svr.py` — adapted polynomial SVR baseline; loads the two supplied LJ Speech mel samples and generates WAV files.
- `colatron_custom_eval/work/colatron_reproduction/run_calibrated.py` — fits a separate Ridge duration estimator and time-stretches generated WAV files.
- `colatron_custom_eval/compare_duration.py` — leave-one-out validation on five independently recorded reference durations.
- `colatron_custom_eval/duration_comparison.csv` — individual fold results.

## Setup
```bash
python -m pip install -r colatron_custom_eval/work/colatron_reproduction/requirements.txt
python colatron_custom_eval/compare_duration.py
cd colatron_custom_eval/work/colatron_reproduction
python run_svr.py --mode baseline --iterations 8
python run_calibrated.py
```

The adapted SVR script also needs the original two mel spectrogram arrays and associated training.txt under `colatron_custom_eval/work/colatron_original/submit_code/training/`. If these binary samples aren't present, use the original source archive supplied with the class assignment.

## Results

On five custom reference recordings, baseline mean absolute duration error was **4.023 s** and leave-one-out calibrated duration prediction error was **0.697 s** (**82.7%** lower). This evaluates **duration prediction only**. The acoustic model was not modified and intelligible speech has not been verified.

## Privacy

Personal voice recordings are intentionally **excluded** because the repository is public. The original project and generated WAVs also require care with provenance: do not claim this reproduces the Stanford model's reported audio quality.
