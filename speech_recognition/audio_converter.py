import numpy as np


class AudioConverter:

    @staticmethod
    def convert(audio_frames: bytes):
        audio_array = np.frombuffer(audio_frames, dtype=np.int16).astype(np.float32)
        return audio_array
