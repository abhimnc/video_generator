"""Generate an LTX-2 video from a source-code repository.

The script scans a repository, writes an LTX-friendly cinematic prompt, and
optionally launches one of the existing ltx_pipelines modules with that prompt.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import logging
import os
import platform
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

LOGGER = logging.getLogger(__name__)

DEFAULT_IGNORES = {
    ".git",
    ".hg",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "node_modules",
    "target",
    "vendor",
}

LANGUAGE_BY_EXTENSION = {
    ".c": "C",
    ".cc": "C++",
    ".cpp": "C++",
    ".cs": "C#",
    ".css": "CSS",
    ".dart": "Dart",
    ".go": "Go",
    ".h": "C/C++ headers",
    ".hpp": "C++ headers",
    ".html": "HTML",
    ".java": "Java",
    ".js": "JavaScript",
    ".jsx": "React JavaScript",
    ".kt": "Kotlin",
    ".md": "Markdown",
    ".php": "PHP",
    ".py": "Python",
    ".rb": "Ruby",
    ".rs": "Rust",
    ".sh": "shell",
    ".sql": "SQL",
    ".swift": "Swift",
    ".toml": "TOML",
    ".ts": "TypeScript",
    ".tsx": "React TypeScript",
    ".yaml": "YAML",
    ".yml": "YAML",
}

IMPORTANT_FILENAMES = {
    "Dockerfile",
    "Makefile",
    "README",
    "README.md",
    "package.json",
    "pyproject.toml",
    "requirements.txt",
    "uv.lock",
}

PIPELINE_MODULES = {
    "distilled": "ltx_pipelines.distilled",
    "one-stage": "ltx_pipelines.ti2vid_one_stage",
    "two-stage": "ltx_pipelines.ti2vid_two_stages",
    "two-stage-hq": "ltx_pipelines.ti2vid_two_stages_hq",
}

PROMPT_STYLES = ("cinematic", "cartoon")
DEFAULT_CARTOON_SUBJECT = (
    "a cheerful young inventor and a tiny helper robot testing a flying paint machine in a bright workshop"
)


@dataclass(frozen=True)
class RepoSummary:
    root: Path
    files_seen: int
    files_used: int
    total_lines: int
    top_languages: list[tuple[str, int]]
    top_dirs: list[tuple[str, int]]
    important_files: list[str]
    keyword_counts: Counter[str]


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Scan a code repo, create an LTX-2 cinematic prompt, and optionally generate a video.",
    )
    parser.add_argument("--repo", type=Path, default=Path.cwd(), help="Repository to visualize. Defaults to cwd.")
    parser.add_argument(
        "--pipeline",
        choices=tuple(PIPELINE_MODULES),
        default="distilled",
        help="Existing LTX pipeline module to run when not using --dry-run.",
    )
    parser.add_argument("--output-path", type=Path, default=Path("outputs/repo-video.mp4"), help="Output MP4 path.")
    parser.add_argument("--prompt-output", type=Path, help="Optional file to save the generated prompt.")
    parser.add_argument("--prompt", help="Use this prompt instead of generating one from the repo.")
    parser.add_argument(
        "--subject",
        help="Cartoon subject to expand into a full LTX prompt, for example 'a kid astronaut on a candy planet'.",
    )
    parser.add_argument(
        "--style",
        choices=PROMPT_STYLES,
        default="cinematic",
        help="Generated prompt style. Use 'cartoon' for bright animated videos.",
    )
    parser.add_argument(
        "--cartoon",
        action="store_true",
        help="Shortcut for --style cartoon.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Only write/log the prompt and command.")
    parser.add_argument("--print-command", action="store_true", help="Log the pipeline command before running it.")
    parser.add_argument("--max-files", type=int, default=160, help="Maximum source files to inspect.")
    parser.add_argument("--max-bytes-per-file", type=int, default=24_000, help="Maximum bytes to read from each file.")
    parser.add_argument("--height", type=int, help="Forwarded to the LTX pipeline.")
    parser.add_argument("--width", type=int, help="Forwarded to the LTX pipeline.")
    parser.add_argument("--num-frames", type=int, help="Forwarded to the LTX pipeline. Must be 8k + 1.")
    parser.add_argument("--frame-rate", type=float, help="Forwarded to the LTX pipeline.")
    parser.add_argument("--seed", type=int, help="Forwarded to the LTX pipeline.")
    parser.add_argument("--gemma-root", type=Path, help="Path to Gemma text encoder files.")
    parser.add_argument("--spatial-upsampler-path", type=Path, help="Path to the LTX spatial upsampler.")
    parser.add_argument("--checkpoint-path", type=Path, help="Full-model checkpoint for non-distilled pipelines.")
    parser.add_argument(
        "--distilled-checkpoint-path",
        type=Path,
        help="Distilled checkpoint for the distilled pipeline.",
    )
    parser.add_argument(
        "--distilled-lora",
        nargs="+",
        metavar=("PATH", "STRENGTH"),
        help="Distilled LoRA arguments for two-stage full-model pipelines.",
    )
    parser.add_argument("--enhance-prompt", action="store_true", help="Forward --enhance-prompt to the pipeline.")
    parser.add_argument(
        "--include-readme",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include README-like files in the repository scan.",
    )
    return parser.parse_known_args()


def should_skip(path: Path, root: Path, include_readme: bool) -> bool:
    rel_parts = path.relative_to(root).parts
    if any(part in DEFAULT_IGNORES for part in rel_parts):
        return True
    if not include_readme and path.name.lower().startswith("readme"):
        return True
    if path.name in IMPORTANT_FILENAMES:
        return False
    return path.suffix not in LANGUAGE_BY_EXTENSION


def iter_candidate_files(root: Path, include_readme: bool) -> list[Path]:
    candidates: list[Path] = []
    for path in root.rglob("*"):
        if path.is_file() and not should_skip(path, root, include_readme):
            candidates.append(path)
    return sorted(candidates, key=lambda item: (len(item.relative_to(root).parts), item.as_posix()))


def read_text_sample(path: Path, max_bytes: int) -> str:
    data = path.read_bytes()[:max_bytes]
    if b"\x00" in data:
        return ""
    return data.decode("utf-8", errors="ignore")


def summarize_repo(root: Path, max_files: int, max_bytes_per_file: int, include_readme: bool) -> RepoSummary:
    root = root.expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Repository not found: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Repository is not a directory: {root}")

    candidates = iter_candidate_files(root, include_readme)
    language_counts: Counter[str] = Counter()
    dir_counts: Counter[str] = Counter()
    keyword_counts: Counter[str] = Counter()
    important_files: list[str] = []
    total_lines = 0
    used = 0

    for path in candidates[:max_files]:
        rel = path.relative_to(root).as_posix()
        text = read_text_sample(path, max_bytes_per_file)
        if not text:
            continue

        used += 1
        total_lines += text.count("\n") + 1
        language_counts[LANGUAGE_BY_EXTENSION.get(path.suffix, path.name)] += 1
        dir_counts[rel.split("/", maxsplit=1)[0]] += 1

        if path.name in IMPORTANT_FILENAMES or path.name.lower().startswith("readme"):
            important_files.append(rel)

        lowered = text.lower()
        for keyword in (
            "api",
            "audio",
            "cli",
            "config",
            "dataset",
            "diffusion",
            "gpu",
            "model",
            "pipeline",
            "prompt",
            "scheduler",
            "test",
            "train",
            "transformer",
            "video",
        ):
            keyword_counts[keyword] += lowered.count(keyword)

    return RepoSummary(
        root=root,
        files_seen=len(candidates),
        files_used=used,
        total_lines=total_lines,
        top_languages=language_counts.most_common(5),
        top_dirs=dir_counts.most_common(5),
        important_files=important_files[:6],
        keyword_counts=keyword_counts,
    )


def join_names(items: list[tuple[str, int]], fallback: str) -> str:
    if not items:
        return fallback
    return ", ".join(name for name, _count in items[:4])


def build_prompt(summary: RepoSummary) -> str:
    languages = join_names(summary.top_languages, "source files")
    dirs = join_names(summary.top_dirs, "project folders")
    dominant_keywords = [name for name, count in summary.keyword_counts.most_common(5) if count > 0]
    motifs = ", ".join(dominant_keywords) if dominant_keywords else "modules, commands, and data flow"
    important = ", ".join(summary.important_files[:3]) if summary.important_files else "the entry files"

    prompt = (
        f"A cinematic visualization of a software repository comes alive as {languages} files assemble into a "
        f"glowing architecture map. Panels labeled {dirs} slide into place, thin streams of light trace imports, "
        f"configuration files, and command paths from {important}, while nodes pulse around {motifs}. The camera "
        "starts in a close macro view of crisp code tokens, pulls back into a three-quarter overhead angle, then "
        "dollies forward through stacked folders as tests, models, and pipeline stages activate in chronological "
        "order. The scene uses precise readable interface details, cool monitor light, warm amber highlights for "
        "active execution, sharp reflections on dark glass, and subtle particles only where data is moving. Near "
        "the end, the repository graph compresses into a luminous render queue, a progress bar fills, and a finished "
        "video thumbnail appears on a workstation display."
    )
    words = prompt.split()
    return " ".join(words[:200])


def build_cartoon_prompt(subject: str = DEFAULT_CARTOON_SUBJECT) -> str:
    subject = subject.strip().rstrip(".")
    prompt = (
        f"A colorful cartoon scene begins with {subject}. The main character moves with quick playful "
        "steps, wide expressive eyes, bouncy gestures, and clear exaggerated reactions while nearby objects wobble, "
        "spin, and spring back with soft squash-and-stretch motion. The character has rounded shapes, clean outlines, "
        "bright clothing, and simple readable facial expressions, with small props that move in time with the action. "
        "The background is full of charming hand-drawn details, layered scenery, and toy-like shapes that stay clear "
        "and uncluttered. The camera starts with a close-up on the character's face, tracks sideways with the action, "
        "then pulls back to a three-quarter wide shot as the scene becomes bigger and funnier. The lighting is warm "
        "and soft, the colors are saturated blue, yellow, coral, green, and white, and the animation looks polished, "
        "family-friendly, and energetic. At the end, a surprising object pops into view and the character reacts with "
        "a joyful cartoon pose."
    )
    words = prompt.split()
    return " ".join(words[:200])


def build_styled_prompt(summary: RepoSummary, style: str, subject: str | None) -> str:
    if style == "cartoon":
        return build_cartoon_prompt(subject or DEFAULT_CARTOON_SUBJECT)
    return build_prompt(summary)


def append_optional(command: list[str], flag: str, value: object | None) -> None:
    if value is not None:
        command.extend([flag, str(value)])


def build_pipeline_command(args: argparse.Namespace, prompt: str, passthrough: list[str]) -> list[str]:
    module = PIPELINE_MODULES[args.pipeline]
    command = [sys.executable, "-m", module]

    if args.pipeline == "distilled":
        append_optional(command, "--distilled-checkpoint-path", args.distilled_checkpoint_path)
    else:
        append_optional(command, "--checkpoint-path", args.checkpoint_path)

    append_optional(command, "--gemma-root", args.gemma_root)
    append_optional(command, "--spatial-upsampler-path", args.spatial_upsampler_path)
    append_optional(command, "--output-path", args.output_path)
    command.extend(["--prompt", prompt])

    if args.distilled_lora:
        command.append("--distilled-lora")
        command.extend(args.distilled_lora)

    for flag in ("height", "width", "num_frames", "frame_rate", "seed"):
        value = getattr(args, flag)
        append_optional(command, f"--{flag.replace('_', '-')}", value)

    if args.enhance_prompt:
        command.append("--enhance-prompt")

    command.extend(passthrough)
    return command


def is_lfs_pointer(path: Path) -> bool:
    try:
        with path.open("rb") as file:
            return file.read(64).startswith(b"version https://git-lfs.github.com/spec")
    except OSError:
        return False


def validate_not_lfs_pointer(path: Path, label: str) -> None:
    if not path.exists():
        raise ValueError(f"{label} does not exist: {path}")
    if is_lfs_pointer(path):
        parent = path.parent
        raise ValueError(
            f"{label} is a Git LFS pointer, not the downloaded model file: {path}. "
            f"Run `git -C {parent} lfs pull` after accepting the model license and logging in."
        )


def validate_model_files(args: argparse.Namespace) -> None:
    if args.pipeline == "distilled" and args.distilled_checkpoint_path is not None:
        validate_not_lfs_pointer(args.distilled_checkpoint_path, "--distilled-checkpoint-path")
    if args.pipeline != "distilled" and args.checkpoint_path is not None:
        validate_not_lfs_pointer(args.checkpoint_path, "--checkpoint-path")
    if args.spatial_upsampler_path is not None:
        validate_not_lfs_pointer(args.spatial_upsampler_path, "--spatial-upsampler-path")
    if args.gemma_root is not None:
        if not args.gemma_root.is_dir():
            raise ValueError(f"--gemma-root does not exist or is not a directory: {args.gemma_root}")
        shards = sorted(args.gemma_root.glob("*.safetensors"))
        if not shards:
            raise ValueError(f"--gemma-root does not contain any .safetensors shards: {args.gemma_root}")
        for shard in shards:
            validate_not_lfs_pointer(shard, "--gemma-root shard")


def validate_cuda_runtime() -> None:
    if platform.system() == "Darwin":
        raise ValueError(
            "LTX-2 generation requires an NVIDIA CUDA GPU. This local machine is macOS, and these pipeline scripts "
            "do not support Mac CPU/MPS generation. Use `--dry-run` here, then run the command on a CUDA machine."
        )

    if importlib.util.find_spec("torch") is None:
        raise ValueError(
            "PyTorch is not installed in this Python environment. Run `uv sync --frozen`, activate `.venv`, "
            "or call `.venv/bin/python scripts/repo_to_video.py ...`."
        )

    torch = importlib.import_module("torch")

    if not torch.cuda.is_available():
        raise ValueError(
            "CUDA is not available in this Python environment. LTX-2 generation requires an NVIDIA CUDA GPU; "
            "Mac CPU/MPS execution is not supported by these pipeline scripts."
        )


def validate_generation_args(args: argparse.Namespace) -> None:
    missing: list[str] = []
    if args.pipeline == "distilled":
        if args.distilled_checkpoint_path is None:
            missing.append("--distilled-checkpoint-path")
    elif args.checkpoint_path is None:
        missing.append("--checkpoint-path")

    if args.gemma_root is None:
        missing.append("--gemma-root")
    if args.spatial_upsampler_path is None and args.pipeline in {"distilled", "two-stage", "two-stage-hq"}:
        missing.append("--spatial-upsampler-path")
    if args.pipeline in {"two-stage", "two-stage-hq"} and not args.distilled_lora:
        missing.append("--distilled-lora")

    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"Missing required generation arguments for --pipeline {args.pipeline}: {joined}")
    validate_model_files(args)
    validate_cuda_runtime()


def pythonpath_with_repo(existing: str | None, repo_root: Path) -> str:
    paths = [
        repo_root / "packages" / "ltx-core" / "src",
        repo_root / "packages" / "ltx-pipelines" / "src",
    ]
    values = [path.as_posix() for path in paths]
    if existing:
        values.append(existing)
    return os.pathsep.join(values)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args, passthrough = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    style = "cartoon" if args.cartoon else args.style

    summary = summarize_repo(args.repo, args.max_files, args.max_bytes_per_file, args.include_readme)
    prompt = args.prompt or build_styled_prompt(summary, style, args.subject)
    LOGGER.info("Generated prompt (%s words): %s", len(prompt.split()), prompt)

    if args.prompt_output:
        args.prompt_output.expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
        args.prompt_output.write_text(prompt + "\n", encoding="utf-8")
        LOGGER.info("Wrote prompt to %s", args.prompt_output)

    command = build_pipeline_command(args, prompt, passthrough)
    if args.print_command or args.dry_run:
        LOGGER.info("Pipeline command: %s", subprocess.list2cmdline(command))

    if args.dry_run:
        return 0

    try:
        validate_generation_args(args)
    except ValueError as error:
        LOGGER.error("Error: %s", error)
        return 2
    args.output_path.expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["PYTHONPATH"] = pythonpath_with_repo(env.get("PYTHONPATH"), repo_root)
    completed = subprocess.run(command, check=False, cwd=repo_root, env=env)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
