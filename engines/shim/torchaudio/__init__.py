"""Minimal torchaudio stand-in for NGC PyTorch (no torchaudio wheel for its torch build).

Covers what Ming-3.0-tts-0.5B touches on the built-in-voice path: load, save,
transforms.Resample. compliance/kaldi.py is torchaudio 2.5.1's own pure-torch file (BSD-2-Clause),
which voice cloning needs for the speaker embedding's fbank features.
"""
import soundfile as sf
import torch

from . import transforms  # noqa: F401
from . import compliance  # noqa: F401

__version__ = "0.0-shim"


def load(path, backend=None):
    data, rate = sf.read(path, dtype="float32", always_2d=True)
    return torch.from_numpy(data.T.copy()), rate


def save(path, waveform, sample_rate):
    sf.write(path, waveform.detach().float().cpu().numpy().T, sample_rate)
