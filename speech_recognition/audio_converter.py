import numpy as np


class AudioConverter:

    @classmethod
    def convert(cls, audio_frames: bytes):
        audio_array = np.frombuffer(audio_frames, dtype=np.int16).astype(np.float32)
        return audio_array
