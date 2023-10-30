from pathlib import Path
import numpy as np

def get_project_root() -> Path:
    return Path(__file__).parent.parent

config_name = get_project_root() / 'src' / 'auction_sim_config.npy'

def is_discretized() -> bool:
    discretized, ctrloosen = np.load(config_name, allow_pickle=True)
    return discretized

def is_ctr_loosen() -> bool:
    discretized, ctrloosen = np.load(config_name, allow_pickle=True)
    return ctrloosen

def set_discretized(discretized: bool) -> None:
    DISCRETIZED = discretized
    CTR_LOOSEN = is_ctr_loosen()
    np.save(config_name, np.array([DISCRETIZED, CTR_LOOSEN]))

def set_ctr_loosen(ctr_loosen: bool) -> None:
    DISCRETIZED = is_discretized()
    CTR_LOOSEN = ctr_loosen
    np.save(config_name, np.array([DISCRETIZED, CTR_LOOSEN]))

def set_discretized_ctrloosen(discretized:bool, ctr_loosen:bool) -> None:
    np.save(config_name, np.array([discretized, ctr_loosen]))




def scaleup_ctr(ctr:float) -> float:
    return ctr*0.7 + 0.3