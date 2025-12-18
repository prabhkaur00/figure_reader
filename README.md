# Figure Reader

This repository contains a prototype “figure reader” pipeline that can digitize Raman spectra deterministically using OpenCV or Matplotlib's image api(TODO), or reason directly with a VLM (Google’s Gemini API). It also includes utilities for comparing digitized curves against ground-truth (GT) text files.

## Repository Layout

- `figure_reader.py` – CLI entry point that orchestrates tools, verification, and Gemini integration.
- `src/` – Python package with the agent (`agent.py`), Gemini client (`gemini_client.py`), tool implementations (`tools.py`), verification helpers (`verification.py`), and digitizers (`opencv_point_extraction.py`, `matplotlib_point_extraction.py`, etc.)..
- `src/prompts/` – Prompt templates for Gemini requests.

## TODOs

- Matplotlib-based digitizer is a stub (`src/matplotlib_point_extraction.py` + `src/tools.py`); implement actual point extraction.
- Verification currently interpolates normalized intensities to compute RMSE/MAE; peak-specific metrics may be preferable for Raman spectra.
- `verification/opencv/raman1_opencv_verify.png` shows that the axis limits supplied to OpenCV (and cached in `llm_params/raman1_limits.json`) are inaccurate. TODO: improve how limits are inferred (e.g., refine the VLM prompt or add deterministic helpers for detecting axes and plot limits) so OpenCV receives accurate ranges without manual tweaking.

## Deterministic Digitization Flow (OpenCV)

1. **Prompting for Limits** – `OpenCVPointExtractor` loads `opencv_limits_prompt.txt` and calls Gemini (via `GEMINI_API_KEY`) with the target image. Gemini returns a JSON describing x/y axis ranges and pixel bounds, which is cached as `llm_params/<image>_limits.json`. Set `--regenerate-prompts` to force a refresh.
2. **OpenCV Extraction** – `opencv_point_extraction.digitize_raman_spectrum` uses the cached JSON to set pixel bounds/axis limits, finds the blue curve via HSV masking, and writes normalized results to `OpenCV/<image>_opencv.csv`.
3. **Verification** – `--mode verify-opencv` digitizes the image, loads the corresponding GT TXT (from `--gt-file` or `--gt-dir`), normalizes both series to [0,1], interpolates the digitized curve onto the GT x-grid, and computes RMSE/MAE + overlap counts. The overlay is saved under `verification/opencv/<image>_verify.png`.


## Environment

Use the provided conda env (`opencvraman`) or create your own with at least:

```bash
conda create -n opencvraman python=3.11
conda activate opencvraman
pip install opencv-python-headless numpy pandas matplotlib google-genai
```

Set your Gemini API key (for VLM/prompt extraction):

```bash
export GEMINI_API_KEY="YOUR_KEY"
```

## Basic Usage

```bash
conda run -n opencvraman python figure_reader.py <image_path> [options]
```

### Common Options

| Flag | Description |
| --- | --- |
| `--mode {auto, opencv, matplotlib, vlm, verify-opencv, verify-matplotlib}` | Select tool directly, auto-select, or run verification + plotting. |
| `--prefer-vlm` | Hint for auto mode to pick the Gemini VLM tool. |
| `--gt-file`, `--gt-dir` | Provide GT TXT path or directory for verification modes. |
| `--gemini-model` | Override Gemini model (default `gemini-2.5-flash`). |

### Examples

Deterministic digitization (OpenCV):

```bash
conda run -n opencvraman python figure_reader.py ground_truth/raman1.png --mode opencv
```

VLM reasoning only (Gemini, requires API key):

```bash
conda run -n opencvraman python figure_reader.py ground_truth/raman1.png --mode vlm
```

Verification with GT overlay/metrics:

```bash
conda run -n opencvraman python figure_reader.py ground_truth/raman1.png --mode verify-opencv --gt-file ground_truth/raman1.txt
```

The verification command digitizes the image via OpenCV, saves the CSV under `OpenCV/`, and writes a normalized overlay + RMSE/MAE summary into `verification/opencv/`.

## OpenCV Axis Limits Prompt

`opencv_limits_prompt.txt` defines the instructions sent to Gemini for extracting plot axes and pixel bounds. Outputs are cached in `llm_params/<image>_limits.json` and reused unless `--regenerate-prompts` is set.

## Ground-Truth Data

Ground-truth spectra (`ground_truth/`) are raw RRUFF-formatted files (Raman shift vs. intensity). Source data can be downloaded from the RRUFF Raman archive: <https://www.rruff.net/zipped_data_files/raman/>.
