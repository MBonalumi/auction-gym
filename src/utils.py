from pathlib import Path

DISCRETIZED = True
CTR_LOOSEN = True

def get_project_root() -> Path:
    return Path(__file__).parent.parent

def is_discretized() -> bool:
    return DISCRETIZED

def is_ctr_loosen() -> bool:
    return CTR_LOOSEN
