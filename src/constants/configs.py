from pathlib import Path
import yaml

_FILE_PATH = Path(__file__).parents[2]
_CONFIG_PATH = _FILE_PATH / "configs" / "config.yaml"
_PROMPT_LIBRARY = _FILE_PATH / "configs" / "prompt_library.yaml"


def _load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


APP_CONFIG = _load_config(_CONFIG_PATH)
PROMPT_LIBRARY = _load_config(_PROMPT_LIBRARY)
