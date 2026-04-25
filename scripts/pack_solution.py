"""
Pack solution source files into solution.json.

Reads configuration from config.toml and packs the appropriate source files
(Triton or CUDA) into a Solution JSON file for submission.
"""

import sys
import json
from pathlib import Path
from typing import Any, Optional

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import tomllib
except ImportError:
    import tomli as tomllib

HAVE_FLASHINFER_BENCH = True
try:
    from flashinfer_bench import BuildSpec, Solution, SourceFile  # type: ignore
except ModuleNotFoundError:
    HAVE_FLASHINFER_BENCH = False
    BuildSpec = None  # type: ignore
    Solution = None  # type: ignore
    SourceFile = None  # type: ignore

def _ordered_source_paths(source_dir: Path, language: str, binding: Optional[str] = None) -> list[Path]:
    """Return source files in a deterministic, submission-friendly order."""
    valid_ext = {".py", ".cu", ".cuh", ".cpp", ".c", ".h", ".hpp", ".cc", ".cxx"}
    files = [p for p in source_dir.iterdir() if p.is_file() and p.suffix.lower() in valid_ext]

    # If using torch C++ extension binding, ignore the TVM-FFI python template.
    if language == "cuda" and binding == "torch":
        files = [p for p in files if p.name != "binding.py"]

    # Match the reference ordering used by the provided GPT solution JSON.
    if language == "cuda":
        preferred = ["kernel.h", "kernel.cu", "main.cpp"]
        rank = {name: i for i, name in enumerate(preferred)}

        def key(p: Path):
            return (rank.get(p.name, 10_000), p.name)

        return sorted(files, key=key)

    # Default: alphabetical
    return sorted(files, key=lambda p: p.name)


# Some installations (e.g. older wheels) may not ship `flashinfer_bench.agents`.
# Provide a local fallback that matches upstream behavior.
if HAVE_FLASHINFER_BENCH:
    try:
        from flashinfer_bench.agents import pack_solution_from_files  # type: ignore
    except ModuleNotFoundError:

        def pack_solution_from_files(
            path: str, spec: BuildSpec, name: str, definition: str, author: str, description: str = ""
        ) -> Solution:
            path_obj = Path(path)
            if not path_obj.exists():
                raise ValueError(f"Path does not exist: {path}")
            if not path_obj.is_dir():
                raise ValueError(f"Path is not a directory: {path}")

            sources = []
            for file_path in _ordered_source_paths(
                path_obj, getattr(spec, "language", "cuda"), getattr(spec, "binding", None)
            ):
                sources.append(SourceFile(path=file_path.name, content=file_path.read_text(encoding="utf-8")))

            if not sources:
                raise ValueError(f"No source files found in directory: {path}")

            return Solution(
                name=name,
                definition=definition,
                author=author,
                description=description or "",
                spec=spec,
                sources=sources,
            )


def _pack_solution_json_without_flashinfer(
    *,
    source_dir: Path,
    language: str,
    name: str,
    definition: str,
    author: str,
    description: str,
    entry_point: str,
    target_hardware: list[str],
    dependencies: list[Any],
    destination_passing_style: Optional[bool],
    binding: Optional[str],
) -> dict[str, Any]:
    spec: dict[str, Any] = {
        "language": language,
        "target_hardware": target_hardware,
        "entry_point": entry_point,
        "dependencies": dependencies,
    }
    if destination_passing_style is not None:
        spec["destination_passing_style"] = destination_passing_style
    if binding is not None:
        spec["binding"] = binding

    sources = []
    for file_path in _ordered_source_paths(source_dir, language, binding):
        sources.append({"path": file_path.name, "content": file_path.read_text(encoding="utf-8")})

    return {
        "name": name,
        "definition": definition,
        "author": author,
        "spec": spec,
        "sources": sources,
        "description": description or "",
    }


def load_config() -> dict:
    """Load configuration from config.toml."""
    config_path = PROJECT_ROOT / "config.toml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "rb") as f:
        return tomllib.load(f)


def pack_solution(output_path: Path = None) -> Path:
    """Pack solution files into a Solution JSON."""
    config = load_config()

    solution_config = config["solution"]
    build_config = config["build"]

    language = build_config["language"]
    entry_point = build_config["entry_point"]
    target_hardware = build_config.get("target_hardware", ["cuda"])
    dependencies = build_config.get("dependencies", [])
    destination_passing_style = build_config.get("destination_passing_style", None)
    binding = build_config.get("binding", None)

    # Determine source directory based on language
    if language == "triton":
        source_dir = PROJECT_ROOT / "solution" / "triton"
    elif language == "cuda":
        source_dir = PROJECT_ROOT / "solution" / "cuda"
    else:
        raise ValueError(f"Unsupported language: {language}")

    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    # Write to output file
    if output_path is None:
        output_path = PROJECT_ROOT / "solution.json"

    if HAVE_FLASHINFER_BENCH:
        # Create build spec (be tolerant to older BuildSpec signatures)
        spec_kwargs: dict[str, Any] = {
            "language": language,
            "target_hardware": target_hardware,
            "entry_point": entry_point,
            "dependencies": dependencies,
        }
        if destination_passing_style is not None:
            spec_kwargs["destination_passing_style"] = destination_passing_style
        if binding is not None:
            spec_kwargs["binding"] = binding

        try:
            spec = BuildSpec(**spec_kwargs)  # type: ignore[misc]
        except TypeError:
            # Fall back to minimal required fields
            spec = BuildSpec(  # type: ignore[misc]
                language=language,
                target_hardware=target_hardware,
                entry_point=entry_point,
            )

        # Pack the solution
        solution = pack_solution_from_files(
            path=str(source_dir),
            spec=spec,
            name=solution_config["name"],
            definition=solution_config["definition"],
            author=solution_config["author"],
            description=solution_config.get("description", ""),
        )

        output_path.write_text(solution.model_dump_json(indent=2))
        print(f"Solution packed: {output_path}")
        print(f"  Name: {solution.name}")
        print(f"  Definition: {solution.definition}")
        print(f"  Author: {solution.author}")
        print(f"  Language: {language}")
    else:
        solution_dict = _pack_solution_json_without_flashinfer(
            source_dir=source_dir,
            language=language,
            name=solution_config["name"],
            definition=solution_config["definition"],
            author=solution_config["author"],
            description=solution_config.get("description", ""),
            entry_point=entry_point,
            target_hardware=target_hardware,
            dependencies=dependencies,
            destination_passing_style=destination_passing_style,
            binding=binding,
        )
        output_path.write_text(json.dumps(solution_dict, indent=2, ensure_ascii=False))
        print(f"Solution packed (no flashinfer_bench installed): {output_path}")
        print(f"  Name: {solution_dict['name']}")
        print(f"  Definition: {solution_dict['definition']}")
        print(f"  Author: {solution_dict['author']}")
        print(f"  Language: {language}")

    return output_path


def main():
    """Entry point for pack_solution script."""
    import argparse

    parser = argparse.ArgumentParser(description="Pack solution files into solution.json")
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Output path for solution.json (default: ./solution.json)"
    )
    args = parser.parse_args()

    try:
        pack_solution(args.output)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
