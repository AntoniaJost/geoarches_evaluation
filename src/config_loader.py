import yaml
import re

class ConfigLoader:
    """
    Loads a YAML config file with support for `${...}` interpolation.
    Interpolation is limited to keys in the `general` section.
    """
    def __init__(self, path):
        # load raw YAML
        with open(path) as f:
            self._raw = yaml.safe_load(f)
        # interpolate variables (e.g., ${general.work_dir})
        self._cfg = self._interpolate(self._raw)

    def _substitute(self, s: str) -> str:
        # build flat mapping for placeholders
        mapping = {}
        def build(prefix, d):
            for k, v in d.items():
                if isinstance(v, (str, int, float, bool)):
                    mapping[f"{prefix}.{k}"] = str(v)
                elif isinstance(v, dict):
                    build(f"{prefix}.{k}", v)
        build("general", self._raw.get("general", {}))

        # replace ${section.key} occurrences
        pattern = re.compile(r"\$\{([^}]+)\}")
        def repl(match):
            key = match.group(1)
            return mapping.get(key, match.group(0))
        return pattern.sub(repl, s)

    def _interpolate(self, obj):
        # recursively interpolate strings in dicts, lists or individual values
        if isinstance(obj, str):
            return self._substitute(obj)
        if isinstance(obj, list):
            return [self._interpolate(v) for v in obj]
        if isinstance(obj, dict):
            return {k: self._interpolate(v) for k, v in obj.items()}
        return obj

    def get(self, section):
        # access section like cfg.get("general")
        return self._cfg.get(section, {})

    def __getattr__(self, name):
        # allow access like cfg.general
        return self._cfg.get(name)
    