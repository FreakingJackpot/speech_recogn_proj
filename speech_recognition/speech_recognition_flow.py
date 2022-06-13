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
        self._streamer_thread = None
        self._recogniser_thread = None

    def start(self) -> None:
        self._streamer_thread = Thread(target=self._sound_streamer.start,
                                       args=(self._sound_queue, self._exit_event))
        self._streamer_thread.start()

        self._recogniser_thread = Thread(target=self._speech_recognizer.start,
                                         args=(self._sound_queue, self._result_queue, self._exit_event))
        self._recogniser_thread.start()

    def stop(self) -> None:
        self._exit_event.set()

    def get_speech(self) -> None | str:
        return self._result_queue.get()
