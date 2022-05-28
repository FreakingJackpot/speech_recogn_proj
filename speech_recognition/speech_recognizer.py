from queue import Queue
from threading import Event

import numpy as np

from speech_recognition.wav2vec import Wav2Vec


class SpeechRecognizer:
    def __init__(self):
        self._wav2vec = Wav2Vec()

    def start(self, sound_queue: Queue, result_queue: Queue, exit_event: Event) -> None:
        while True:
            if exit_event.is_set():
                break
            if not sound_queue.empty():
                audio_frames = sound_queue.get()
                if audio_frames:
                    audio_array = np.frombuffer(audio_frames, dtype=np.int16).astype(np.float32)
                    text = self._wav2vec.array_to_text(audio_array)
                    result_queue.put(text)
