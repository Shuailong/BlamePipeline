import os
import sys
from pathlib import PosixPath

if sys.version_info < (3, 5):
    raise RuntimeError('Supports Python 3.5 or higher.')

DATA_DIR = (
    os.getenv('BLAME_DATA') or
    os.path.join(PosixPath(__file__).absolute().parents[1].as_posix(), 'data')
)
