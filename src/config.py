from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_config(config_path: str | Path | None = None) -> dict:
    """Load project configuration from a YAML file."""
    if config_path is None:
        config_path = PROJECT_ROOT / "config.yaml"
    else:
        config_path = Path(config_path)

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Resolve relative data paths against project root
    if "data_paths" in cfg:
        for key, value in cfg["data_paths"].items():
            if isinstance(value, str) and not Path(value).is_absolute():
                cfg["data_paths"][key] = str(PROJECT_ROOT / value)

    return cfg
