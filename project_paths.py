from pathlib import Path


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
JSON_RESULTS_DIR = RESULTS_DIR / "json"
PLOT_RESULTS_DIR = RESULTS_DIR / "plots"
ARCHIVE_DIR = ROOT / "archive"
LEGACY_DIR = ARCHIVE_DIR / "legacy"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def default_json_path(filename: str) -> Path:
    return ensure_dir(JSON_RESULTS_DIR) / filename


def resolve_json_path(path: Path) -> Path:
    if not path.is_absolute() and path.parent == Path("."):
        path = default_json_path(path.name)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
    return path


def default_plot_path(filename: str) -> Path:
    path = ensure_dir(PLOT_RESULTS_DIR) / Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def resolve_plot_path(path: Path) -> Path:
    if not path.is_absolute() and path.parent == Path("."):
        path = default_plot_path(path.name)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
    return path


def repo_relative_path(path: Path | str) -> str:
    path = Path(path)
    if not path.is_absolute():
        return path.as_posix()
    try:
        return path.resolve(strict=False).relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()
