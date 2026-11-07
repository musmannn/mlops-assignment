"""Task 1.1 - print exact library versions and whether TensorFlow sees a GPU.

    python labs/env_check.py
"""

import getpass
import platform
import socket
import sys
from datetime import datetime
from importlib.metadata import version

import tensorflow as tf

print(f"user={getpass.getuser()} host={socket.gethostname()} time={datetime.now():%Y-%m-%d %H:%M:%S}")
print(f"python      {sys.version.split()[0]} ({sys.executable})")
print(f"platform    {platform.platform()}")
for pkg in ["tensorflow-cpu", "keras", "keras-tuner", "mlflow", "dagshub", "dvc", "numpy", "pandas",
            "pillow", "scikit-learn", "matplotlib", "flask", "gunicorn"]:
    print(f"{pkg:16s}{version(pkg)}")
gpus = tf.config.list_physical_devices("GPU")
print(f"GPUs visible to TensorFlow: {gpus or 'none (CPU only)'}")
print(f"built with CUDA: {tf.test.is_built_with_cuda()}")
