import torch
import torch.nn.functional as F


class Resample(torch.nn.Module):
    def __init__(self, orig_freq, new_freq):
        super().__init__()
        self.orig, self.new = orig_freq, new_freq

    def forward(self, waveform):
        if self.orig == self.new:
            return waveform
        n = int(round(waveform.shape[-1] * self.new / self.orig))
        return F.interpolate(waveform.unsqueeze(0), size=n, mode="linear", align_corners=False).squeeze(0)
