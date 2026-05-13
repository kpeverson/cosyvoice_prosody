import librosa
import numpy as np
import scipy.signal as signal
import torch
import torchaudio


class GlottalExtractor:
    """
    Frame-based LPC inverse filtering to approximate the glottal source signal.
    Matches the extraction used to train the prosodyvec prosody encoder:
      sr=16000, window=25ms, stride=10ms, order=16, Hamming window, LPF at 1000 Hz.
    """

    def __init__(
        self,
        sr: int = 16000,
        lpc_window_size: float = 0.025,
        lpc_window_stride: float = 0.010,
        lpc_order: int = 16,
        lpc_window: str = "hamming",
        lpf_cutoff: float = 1000.0,
        lpf_order: int = 4,
        energy_threshold: float = 1e-4,
    ):
        self.sr = sr
        self.lpc_window_size = int(sr * lpc_window_size)
        self.lpc_window_stride = int(sr * lpc_window_stride)
        self.lpc_order = lpc_order
        self.lpc_window = lpc_window
        self.lpf_cutoff = lpf_cutoff
        self.lpf_order = lpf_order
        self.energy_threshold = energy_threshold

        if self.lpf_cutoff is not None:
            self._lpf_sos = signal.butter(
                self.lpf_order,
                self.lpf_cutoff,
                "low",
                fs=self.sr,
                output="sos",
            )

    def _inverse_filter(self, x_frame: np.ndarray, a: np.ndarray) -> np.ndarray:
        if np.sum(x_frame ** 2) < self.energy_threshold:
            return x_frame
        # a = [1, a1, ..., ap] from librosa.lpc
        # prediction: lfilter([0, -a1, ..., -ap], [1], x_frame)
        x_hat = signal.lfilter(np.hstack([[0], -1 * a[1:]]), [1], x_frame)
        return x_frame - x_hat

    def extract(self, x: torch.Tensor) -> np.ndarray:
        """
        Args:
            x: 1-D float32 tensor at self.sr Hz.
        Returns:
            glottal source as float32 numpy array of the same length.
        """
        x_np = x.numpy().astype(np.float64)
        glottal = np.zeros_like(x_np)

        frames = librosa.util.frame(
            x_np,
            frame_length=self.lpc_window_size,
            hop_length=self.lpc_window_stride,
        ).T

        if self.lpc_window == "hamming":
            window = np.hamming(self.lpc_window_size)
        else:
            raise ValueError(f"Unsupported window: {self.lpc_window}")

        for i, frame in enumerate(frames):
            frame_w = frame * window
            a = librosa.lpc(frame_w, order=self.lpc_order)
            g = self._inverse_filter(frame_w, a)
            start = i * self.lpc_window_stride
            glottal[start : start + self.lpc_window_size] += g

        if self.lpf_cutoff is not None:
            glottal = signal.sosfiltfilt(self._lpf_sos, glottal)

        return glottal.astype(np.float32)


_default_extractor: GlottalExtractor = None


def get_default_extractor() -> GlottalExtractor:
    global _default_extractor
    if _default_extractor is None:
        _default_extractor = GlottalExtractor()
    return _default_extractor


def extract_glottal_source(
    audio: torch.Tensor,
    src_sample_rate: int,
    target_sample_rate: int = 16000,
    extractor: GlottalExtractor = None,
) -> torch.Tensor:
    """
    Resample audio to 16 kHz, extract glottal source, return as float32 tensor.

    Args:
        audio:            [T] or [1, T] float32 tensor.
        src_sample_rate:  sample rate of `audio`.
        target_sample_rate: sample rate expected by the extractor (default 16000).
        extractor:        GlottalExtractor instance; uses module-level default if None.

    Returns:
        [T'] float32 tensor at target_sample_rate.
    """
    if audio.dim() == 2:
        audio = audio.squeeze(0)
    if src_sample_rate != target_sample_rate:
        audio = torchaudio.functional.resample(
            audio.unsqueeze(0), src_sample_rate, target_sample_rate
        ).squeeze(0)

    if extractor is None:
        extractor = get_default_extractor()

    glottal = extractor.extract(audio.cpu().float())
    return torch.from_numpy(glottal)
