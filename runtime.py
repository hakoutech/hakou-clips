"""Make optional project-local tools available without changing the system PATH."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def configure():
    bins = sorted((ROOT / '.tools' / 'ffmpeg').glob('*/bin'))
    if bins:
        os.environ['PATH'] = os.pathsep.join([str(bins[-1]), os.environ.get('PATH', '')])
    os.environ.setdefault('HF_HOME', str(ROOT / '.cache' / 'huggingface'))
    os.environ.setdefault('PIP_CACHE_DIR', str(ROOT / '.cache' / 'pip'))

