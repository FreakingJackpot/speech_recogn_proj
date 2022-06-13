from queue import Queue
from threading import Event

from speech_recognition.audio_converter import AudioConverter
from speech_recognition.wav2vec import Wav2Vec


class SpeechRecognizer:
    _converter = AudioConverter

    def __init__(self):
        self._wav2vec = Wav2Vec()

    def start(self, sound_queue: Queue, result_queue: Queue, exit_event: Event) -> None:
        while True:
            if exit_event.is_set():
                break
            if not sound_queue.empty():
                audio_frames = sound_queue.get()
                if audio_frames:
                    text = self._recognize(audio_frames)
                    result_queue.put(text)

    def _recognize(self, audio_frames: bytes) -> str:
        audio_array = self._converter.convert(audio_frames)
        text = self._wav2vec.array_to_text(audio_array)
        return text
