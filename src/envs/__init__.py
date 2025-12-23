from .env_2d import Simple2DGridENV
from .env_3d import Simple3DGridENV

__all__ = ["Simple2DGridENV", "Simple3DGridENV"]

_ENVS = {
    "2d": Simple2DGridENV,
    "3d": Simple3DGridENV
}

def get_env(env_name, **kwargs):
    if env_name in _ENVS:
        return _ENVS[env_name](**kwargs)
    else:
        raise ValueError(f"Environment '{env_name}' not found.")