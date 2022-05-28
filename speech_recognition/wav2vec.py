from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch

from config import settings

DEFAULT_REPOSITORY = 'jonatasgrosman/wav2vec2-large-xlsr-53-russian'


class Wav2Vec:
    def __init__(self):
        repository = settings.get('repository_name', DEFAULT_REPOSITORY)

        self._processor = Wav2Vec2Processor.from_pretrained(repository)
        self._model = Wav2Vec2ForCTC.from_pretrained(repository)

        print('я запустилься!')

    def array_to_text(self, audio_array) -> str:
        if not len(audio_array):
            return ""

        inputs = self._processor(torch.tensor(audio_array), sampling_rate=16_000, return_tensors="pt", padding=True)

        with torch.no_grad():
            logits = self._model(inputs.input_values).logits

        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = self._processor.batch_decode(predicted_ids)[0]
        return transcription.lower()
