from threading import Thread, Event
from queue import Queue


from speech_recognition.sound_streamer import SoundStreamer
from speech_recognition.speech_recognizer import SpeechRecognizer


class SpeechRecognitionFlow:

    def __init__(self):
        self._sound_streamer = SoundStreamer()
        self._speech_recognizer = SpeechRecognizer()
        self._exit_event = Event()
        self._result_queue = Queue()
        self._sound_queue = Queue()
        self._streamer_process = None
        self._recogniser_process = None

    def start(self) -> None:
        self._streamer_process = Thread(target=self._sound_streamer.start,
                                        args=(self._sound_queue, self._exit_event))
        self._streamer_process.start()

        self._recogniser_process = Thread(target=self._speech_recognizer.start,
                                          args=(self._sound_queue, self._result_queue, self._exit_event))
        self._recogniser_process.start()

    def stop(self) -> None:
        self._exit_event.set()
        self._streamer_process.join()
        self._recogniser_process.join()

    def get_speech(self) -> None | str:
        return self._result_queue.get()
