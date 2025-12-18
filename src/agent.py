from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


@dataclass
class FigureReaderRequest:
    image_path: Path
    mode: str
    regenerate_prompts: bool = False
    prefer_vlm: bool = False


@dataclass
class FigureReaderResult:
    tool_name: str
    outputs: Dict[str, Optional[Path]] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class FigureReaderTool:
    name: str

    def run(self, request: FigureReaderRequest) -> FigureReaderResult:
        raise NotImplementedError


class FigureReaderAgent:
    def __init__(self, tools: Iterable[FigureReaderTool]):
        self._tools = {tool.name: tool for tool in tools}

    def choose_tool(self, request: FigureReaderRequest) -> FigureReaderTool:
        if request.mode != "auto":
            try:
                return self._tools[request.mode]
            except KeyError as exc:
                raise KeyError(f"Unknown tool '{request.mode}'") from exc

        if request.prefer_vlm and "vlm" in self._tools:
            return self._tools["vlm"]
        if "opencv" in self._tools:
            return self._tools["opencv"]
        if "matplotlib" in self._tools:
            return self._tools["matplotlib"]
        raise RuntimeError("No tools registered for auto mode.")

    def run(self, request: FigureReaderRequest) -> FigureReaderResult:
        tool = self.choose_tool(request)
        return tool.run(request)
