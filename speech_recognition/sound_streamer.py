from queue import Queue
from threading import Event

import webrtcvad
import pyaudio

from config import settings

DEFAULT_AUDIO_LENGTH = 3
DEFAULT_FRAME_DURATION = 20


class SoundStreamer:
    _format = pyaudio.paInt16
    _channels = 1
    _sample_rate = 16000

    def __init__(self):
        self._audio_length = settings.get('audio_length', DEFAULT_AUDIO_LENGTH)
        self._device_name = settings.get('device_name')
        self._frame_duration = settings.get('frame_duration', DEFAULT_FRAME_DURATION)
        self._chunk = int(self._sample_rate * self._frame_duration / 1000)

    def start(self, sound_queue: Queue, exit_event: Event) -> None:
        vad = webrtcvad.Vad(1)
        audio = pyaudio.PyAudio()
        stream = self._init_stream(audio)

        frames = b''
        while True:
            if exit_event.is_set():
                break

            frame = stream.read(self._chunk)
            if vad.is_speech(frame, self._sample_rate):
                frames += frame
            elif frames:
                sound_queue.put(frames)
                frames = b''

        stream.stop_stream()
        stream.close()
        audio.terminate()

    def _init_stream(self, audio) -> pyaudio.Stream:
        device_index = self._get_input_device_index(audio)

        stream = audio.open(input_device_index=device_index, format=self._format, channels=self._channels,
                            rate=self._sample_rate, input=True, frames_per_buffer=self._chunk)
        return stream

    def _get_input_device_index(self, audio: pyaudio.PyAudio) -> int:
        device_name__index = self._get_record_devices_dict(audio)

        if self._device_name:
            return device_name__index[self._device_name]

        return [value for value in device_name__index.values()][0]

    def _get_record_devices_dict(self, audio: pyaudio.PyAudio):
        info = audio.get_host_api_info_by_index(0)
        numdevices = info.get('deviceCount')

        device_name__index = {}
        for i in range(0, numdevices):
            if (audio.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
                name = audio.get_device_info_by_host_api_device_index(0, i).get('name')
                device_name__index[name] = i
        return device_name__index
