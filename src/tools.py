from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

from src.opencv_point_extraction import digitize_raman_spectrum

from .agent import FigureReaderRequest, FigureReaderResult, FigureReaderTool
from .gemini_client import GeminiClient, clean_prompt_text


class OpenCVPointExtractor(FigureReaderTool):
    name = "opencv"

    def __init__(
        self,
        *,
        gemini_client: Optional[GeminiClient],
        prompt_text: str,
        params_root: Path,
        csv_root: Path,
    ):
        self._gemini_client = gemini_client
        self._prompt_text = prompt_text
        self._params_root = params_root
        self._csv_root = csv_root

    def run(self, request: FigureReaderRequest) -> FigureReaderResult:
        params_path = self._params_root / f"{request.image_path.stem}_limits.json"
        csv_path = self._csv_root / f"{request.image_path.stem}_opencv.csv"
        params_path.parent.mkdir(parents=True, exist_ok=True)
        csv_path.parent.mkdir(parents=True, exist_ok=True)

        needs_prompt = request.regenerate_prompts or not params_path.exists()
        if needs_prompt:
            if self._gemini_client is None:
                raise RuntimeError(
                    "Gemini client unavailable; cannot build OpenCV limits prompt."
                )
            llm_json = self._gemini_client.generate_json(
                self._prompt_text,
                image_path=request.image_path,
                temperature=0.0,
            )
            json.loads(llm_json)
            params_path.write_text(llm_json, encoding="utf-8")

        digitize_raman_spectrum(
            image_path=request.image_path,
            llm_params_json=params_path,
            output_csv=csv_path,
        )
        metadata = {"params_path": params_path, "csv_path": csv_path}
        if needs_prompt:
            metadata["prompt_refreshed"] = True
        return FigureReaderResult(
            tool_name=self.name,
            outputs={"csv": csv_path, "llm_params": params_path},
            metadata=metadata,
        )


class MatplotlibImageExtractor(FigureReaderTool):
    name = "matplotlib"

    def run(self, request: FigureReaderRequest) -> FigureReaderResult:  # pragma: no cover
        notes = (
            "Matplotlib-based digitizer is not implemented yet. "
            "Stub included to reserve the tool interface."
        )
        return FigureReaderResult(
            tool_name=self.name,
            outputs={},
            metadata={"status": "not_implemented", "notes": notes},
        )


class GeminiVlmExtractor(FigureReaderTool):
    name = "vlm"

    def __init__(
        self,
        *,
        gemini_client: Optional[GeminiClient],
        output_root: Path,
        prompt_text: str,
        temperature: float = 0.1,
    ):
        if gemini_client is None:
            raise RuntimeError("Gemini client is required for the VLM tool.")
        self._gemini_client = gemini_client
        self._output_root = output_root
        self._output_root.mkdir(parents=True, exist_ok=True)
        self._prompt_text = prompt_text
        self._temperature = temperature

    def run(self, request: FigureReaderRequest) -> FigureReaderResult:
        json_path = self._output_root / f"{request.image_path.stem}_vlm.json"
        raw = self._gemini_client.generate_json(
            self._prompt_text,
            image_path=request.image_path,
            temperature=self._temperature,
        )
        json.loads(raw)
        json_path.write_text(raw, encoding="utf-8")
        return FigureReaderResult(
            tool_name=self.name,
            outputs={"json": json_path},
            metadata={"json_path": json_path},
        )


def build_tools(args, gemini_client: Optional[GeminiClient]) -> List[FigureReaderTool]:
    prompt_template_path = Path(args.opencv_prompt_template)
    if not prompt_template_path.exists():
        raise FileNotFoundError(f"OpenCV prompt template missing: {prompt_template_path}")

    params_root = Path(args.llm_params_dir)
    csv_root = Path(args.opencv_output_dir)
    params_root.mkdir(parents=True, exist_ok=True)
    csv_root.mkdir(parents=True, exist_ok=True)

    opencv_prompt_text = load_prompt_text(prompt_template_path)

    tools: List[FigureReaderTool] = []
    tools.append(
        OpenCVPointExtractor(
            gemini_client=gemini_client,
            prompt_text=opencv_prompt_text,
            params_root=params_root,
            csv_root=csv_root,
        )
    )
    tools.append(MatplotlibImageExtractor())

    if gemini_client is not None:
        vlm_prompt_path = Path(args.vlm_prompt_path)
        if not vlm_prompt_path.exists():
            raise FileNotFoundError(f"VLM prompt file missing: {vlm_prompt_path}")
        vlm_prompt = load_prompt_text(vlm_prompt_path)
        tools.append(
            GeminiVlmExtractor(
                gemini_client=gemini_client,
                output_root=Path(args.vlm_output_dir),
                prompt_text=vlm_prompt,
                temperature=args.vlm_temperature,
            )
        )
    return tools


def load_prompt_text(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    return clean_prompt_text(text)
