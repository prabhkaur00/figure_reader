from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from src.agent import FigureReaderAgent, FigureReaderRequest
from src.gemini_client import GeminiClient, DEFAULT_GEMINI_MODEL
from src.tools import build_tools
from src.verification import verify_digitized_vs_gt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Figure Reader orchestration agent.")
    parser.add_argument("image", type=Path, help="Plot image to analyze.")
    parser.add_argument(
        "--mode",
        default="auto",
        choices=[
            "auto",
            "opencv",
            "matplotlib",
            "vlm",
            "verify-opencv",
            "verify-matplotlib",
        ],
        help="Selects the tool explicitly, auto, or verify-* modes that compare against GT.",
    )
    parser.add_argument(
        "--prefer-vlm",
        action="store_true",
        help="Hint for auto mode to select the Gemini VLM tool.",
    )
    parser.add_argument(
        "--regenerate-prompts",
        action="store_true",
        help="Force-refresh of OpenCV limit prompts even if cached.",
    )
    parser.add_argument(
        "--opencv-prompt-template",
        default="src/prompts/opencv_limits_prompt.txt",
        help="Prompt template used to request OpenCV limits from Gemini.",
    )
    parser.add_argument(
        "--llm-params-dir",
        default="llm_params",
        help="Directory storing cached OpenCV limit JSON outputs.",
    )
    parser.add_argument(
        "--opencv-output-dir",
        default="OpenCV",
        help="Directory where OpenCV CSV digitizations are saved.",
    )
    parser.add_argument(
        "--vlm-output-dir",
        default="Gemini",
        help="Directory where Gemini reasoning JSON files are saved.",
    )
    parser.add_argument(
        "--vlm-prompt-path",
        default="src/prompts/vlm_prompt_default.txt",
        help="Prompt file for the Gemini reasoning tool.",
    )
    parser.add_argument(
        "--vlm-temperature",
        default=0.1,
        type=float,
        help="Temperature for the Gemini VLM tool.",
    )
    parser.add_argument(
        "--gemini-model",
        default=DEFAULT_GEMINI_MODEL,
        help="Gemini API model name used for both prompt extraction and VLM reasoning.",
    )
    parser.add_argument(
        "--gemini-max-output-tokens",
        default=2048,
        type=int,
        help="Max tokens for Gemini responses.",
    )
    parser.add_argument(
        "--gt-file",
        type=Path,
        default=None,
        help="Optional explicit GT text file for verification.",
    )
    parser.add_argument(
        "--gt-dir",
        type=Path,
        default=Path("GT/all_pts"),
        help="Directory to search for GT text files when --gt-file is not set.",
    )
    parser.add_argument(
        "--verify-output-dir",
        type=Path,
        default=Path("Verification"),
        help="Directory where verification overlays/metrics are saved.",
    )
    parser.add_argument(
        "--disable-verify-interp",
        action="store_true",
        help="Skip plotting the interpolated prediction on GT x-grid during verification.",
    )
    return parser.parse_args()


def resolve_gt_path(image_path: Path, args: argparse.Namespace) -> Path:
    if args.gt_file is not None:
        explicit = Path(args.gt_file)
        if not explicit.exists():
            raise FileNotFoundError(f"GT file not found: {explicit}")
        return explicit

    candidates = []
    stem = image_path.stem
    gt_dir = Path(args.gt_dir)
    candidates.append(gt_dir / f"{stem}.txt")
    candidates.append(gt_dir / f"{stem}_300.txt")
    gt_root = Path("GT")
    candidates.append(gt_root / f"{stem}_300.txt")
    candidates.append(gt_root / f"{stem}.txt")

    seen = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Could not locate GT file for '{stem}'. Provide --gt-file or update --gt-dir."
    )


def main():
    args = parse_args()
    image_path: Path = args.image
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    mode_arg = args.mode
    verify_mode = mode_arg.startswith("verify-")
    effective_mode = mode_arg.split("-", 1)[1] if verify_mode else mode_arg

    gemini_client: Optional[GeminiClient] = None
    try:
        gemini_client = GeminiClient(
            model=args.gemini_model,
            max_output_tokens=args.gemini_max_output_tokens,
        )
    except RuntimeError as exc:
        if effective_mode in {"vlm"} or args.prefer_vlm:
            raise
        print(f"[warning] Gemini client disabled: {exc}")

    tools = build_tools(args, gemini_client)
    agent = FigureReaderAgent(tools)

    request = FigureReaderRequest(
        image_path=image_path,
        mode=effective_mode,
        regenerate_prompts=args.regenerate_prompts,
        prefer_vlm=args.prefer_vlm,
    )
    result = agent.run(request)
    print(f"[agent] Tool used: {result.tool_name}")
    for label, path in result.outputs.items():
        if path:
            print(f"[agent] {label}: {path}")
    if result.metadata:
        print(f"[agent] metadata: {result.metadata}")

    if verify_mode:
        gt_path = resolve_gt_path(image_path, args)
        csv_output = result.outputs.get("csv")
        if csv_output is None:
            raise RuntimeError(
                "Selected tool does not produce a CSV output; cannot run verification."
            )
        verify_dir = Path(args.verify_output_dir) / result.tool_name
        verify_dir.mkdir(parents=True, exist_ok=True)
        report = verify_digitized_vs_gt(
            digitized_csv_path=csv_output,
            gt_txt_path=gt_path,
            out_dir=verify_dir,
            image_stem=image_path.stem,
            tool_name=result.tool_name,
            show_interp=not args.disable_verify_interp,
        )
        print("[verify] overlay:", report["overlay_path"])
        print("[verify] RMSE (normalized):", report["rmse"])
        print("[verify] MAE  (normalized):", report["mae"])
        print("[verify] overlap points:", report["overlap_points"])


if __name__ == "__main__":
    main()
