from __future__ import annotations

import mimetypes
import os
import warnings
from pathlib import Path
from typing import Any, Optional


DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"


class GeminiClient:
    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        model: str = DEFAULT_GEMINI_MODEL,
        max_output_tokens: int = 2048,
    ):
        api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("Set GEMINI_API_KEY to call the Gemini API.")

        self._backend = None
        self._client = None
        self._model_name = model
        self._legacy_model = None
        self._genai_types = None
        self._max_output_tokens = max_output_tokens

        google_genai = self._try_import_google_genai()
        if google_genai is not None:
            self._backend = "google.genai"
            self._client = google_genai.Client(api_key=api_key)
            self._genai_types = google_genai.types
            return

        legacy_mod = self._try_import_google_generativeai()
        if legacy_mod is None:
            raise RuntimeError(
                "Missing Gemini client libraries. Install `google-genai` (preferred) "
                "or `google-generativeai`."
            )

        warnings.warn(
            "google-generativeai is deprecated. Please install google-genai.",
            DeprecationWarning,
            stacklevel=2,
        )
        legacy_mod.configure(api_key=api_key)
        self._legacy_model = legacy_mod.GenerativeModel(model)
        self._backend = "google.generativeai"
        self._legacy_module = legacy_mod

    def generate_json(
        self,
        prompt_text: str,
        *,
        image_path: Optional[Path] = None,
        temperature: float = 0.0,
        response_mime_type: str = "application/json",
    ) -> str:
        if self._backend == "google.genai":
            contents = self._build_genai_contents(prompt_text, image_path)
            response = self._client.models.generate_content(  # type: ignore[union-attr]
                model=self._model_name,
                contents=contents,
                config={
                    "temperature": temperature,
                    "response_mime_type": response_mime_type,
                    "max_output_tokens": self._max_output_tokens,
                },
            )
        elif self._backend == "google.generativeai":
            payload = [{"text": prompt_text}]
            if image_path is not None:
                mime = _guess_mime_type(image_path)
                data = image_path.read_bytes()
                payload.append({"mime_type": mime, "data": data})
            response = self._legacy_model.generate_content(  # type: ignore[union-attr]
                [
                    {
                        "role": "user",
                        "parts": payload,
                    }
                ],
                generation_config={
                    "temperature": temperature,
                    "response_mime_type": response_mime_type,
                    "max_output_tokens": self._max_output_tokens,
                },
            )
        else:  # pragma: no cover
            raise RuntimeError("Gemini client is not configured.")

        text = _extract_text(response)
        return _strip_code_fences(text.strip())

    @staticmethod
    def _try_import_google_genai():
        try:
            from google import genai  # type: ignore

            return genai
        except ImportError:
            return None

    def _build_genai_contents(
        self,
        prompt_text: str,
        image_path: Optional[Path],
    ):
        parts = [
            self._genai_types.Part.from_text(text=prompt_text)  # type: ignore[union-attr]
        ]
        if image_path is not None:
            mime = _guess_mime_type(image_path)
            data = image_path.read_bytes()
            parts.append(
                self._genai_types.Part.from_bytes(  # type: ignore[union-attr]
                    data=data,
                    mime_type=mime,
                )
            )
        return [
            self._genai_types.Content(  # type: ignore[union-attr]
                role="user",
                parts=parts,
            )
        ]

    @staticmethod
    def _try_import_google_generativeai():
        try:
            import google.generativeai as genai  # type: ignore

            return genai
        except ImportError:
            return None


def clean_prompt_text(prompt: str) -> str:
    return "\n".join(line.rstrip() for line in prompt.strip().splitlines())


def _guess_mime_type(path: Path) -> str:
    mime, _ = mimetypes.guess_type(path.as_posix())
    return mime or "image/png"


def _strip_code_fences(text: str) -> str:
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines)
    return text.strip()


def _extract_text(response: Any) -> str:
    for attr in ("text", "output_text", "result"):
        val = getattr(response, attr, None)
        if isinstance(val, str) and val.strip():
            return val

    candidates = getattr(response, "candidates", None)
    texts = []
    if candidates:
        for cand in candidates:
            parts = getattr(cand, "content", None)
            if parts is not None:
                parts = getattr(parts, "parts", None) or getattr(cand, "parts", None)
            else:
                parts = getattr(cand, "parts", None)
            if parts is None:
                continue
            for part in parts:
                text = None
                if isinstance(part, str):
                    text = part
                elif isinstance(part, dict):
                    text = part.get("text")
                else:
                    text = getattr(part, "text", None)
                if text:
                    texts.append(text)
    if texts:
        return "\n".join(texts)

    raise RuntimeError("Could not extract text from Gemini response.")
