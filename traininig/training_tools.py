import re
import json
from os.path import exists
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import librosa
import torch
import numpy as np
from datasets import load_dataset, load_metric, Audio
from transformers import Wav2Vec2CTCTokenizer, AutoProcessor, Trainer, Wav2Vec2FeatureExtractor, Wav2Vec2Processor, \
    Wav2Vec2ForCTC, TrainingArguments


class DatasetFactory:
    CHARS_TO_IGNORE = [
        ",", "?", "¿", ".", "!", "¡", ";", "；", ":", '""', "%", '"', "�", "ʿ", "·", "჻", "~", "՞",
        "؟", "،", "।", "॥", "«", "»", "„", "“", "”", "「", "」", "‘", "’", "《", "》", "(", ")", "[", "]",
        "{", "}", "=", "`", "_", "+", "<", ">", "…", "–", "°", "´", "ʾ", "‹", "›", "©", "®", "—", "→",
        "。",
        "、", "﹂", "﹁", "‧", "～", "﹏", "，", "｛", "｝", "（", "）", "［", "］", "【", "】", "‥", "〽",
        "『", "』", "〝", "〟", "⟨", "⟩", "〜", "：", "！", "？", "♪", "؛", "/", "\\", "º", "−", "^", "ʻ", "ˆ"
    ]
    AUDIO_LENGTH = 3

    def __init__(self):
        self.train_ds = load_dataset("mozilla-foundation/common_voice_9_0", "ru", split="train+validation+other",
                                     use_auth_token=True)
        self.test_ds = load_dataset("mozilla-foundation/common_voice_9_0", "ru", split="test", use_auth_token=True)

        self.chars_to_remove_regex = f"[{re.escape(''.join(self.CHARS_TO_IGNORE))}]"

    def create(self):
        train_ds = self.train_ds.map(self._remove_special_characters)
        test_ds = self.test_ds.map(self._remove_special_characters)

        train_ds = train_ds.cast_column("audio", Audio(sampling_rate=16_000))
        test_ds = test_ds.cast_column("audio", Audio(sampling_rate=16_000))

        train_ds = train_ds.filter(self._is_audio_in_length_range)

        return train_ds, test_ds

    def _remove_special_characters(self, batch):
        batch["sentence"] = re.sub(self.chars_to_remove_regex, '', batch["sentence"]).lower()
        batch["sentence"] = re.sub('[-]', ' ', batch["sentence"]).lower()
        return batch

    def _is_audio_in_length_range(self, batch):
        audio = batch["audio"]
        return False if librosa.get_duration(y=audio["array"], sr=audio["sampling_rate"]) > self.AUDIO_LENGTH else True


@dataclass
class DataCollatorCTCWithPadding:
    processor: AutoProcessor
    padding: Union[bool, str] = "longest"
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                pad_to_multiple_of=self.pad_to_multiple_of_labels,
                return_tensors="pt",
            )

        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels

        return batch


class ModelTrainer:
    def __init__(self):
        dataset_factory = DatasetFactory()
        self.train_ds, self.test_ds = dataset_factory.create()
        self.processor = self._get_processor()
        self.data_collator = DataCollatorCTCWithPadding(processor=self.processor, padding=True)

        self.train_ds = self.train_ds.map(self._prepare_dataset, remove_columns=self.train_ds.column_names)
        self.test_ds = self.test_ds.map(self._prepare_dataset, remove_columns=self.test_ds.column_names)

        self.wer_metric = load_metric("wer")

    def _get_processor(self):
        tokenizer = self._get_tokenizer()
        feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0,
                                                     do_normalize=True, return_attention_mask=True)
        processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

        return processor

    def _get_tokenizer(self):
        if not exists('vocab.json'):
            self._create_vocabruary()
        tokenizer = Wav2Vec2CTCTokenizer.from_pretrained("./", unk_token="[UNK]", pad_token="[PAD]",
                                                         word_delimiter_token="|")

        return tokenizer

    def _create_vocabruary(self):
        vocab_train = self.train_ds.map(self._extract_all_chars, batched=True, batch_size=-1, keep_in_memory=True,
                                        remove_columns=self.train_ds.column_names)
        vocab_test = self.test_ds.map(self._extract_all_chars, batched=True, batch_size=-1, keep_in_memory=True,
                                      remove_columns=self.test_ds.column_names)

        vocab_list = list(set(vocab_train["vocab"][0]) | set(vocab_test["vocab"][0]))

        vocab_dict = {v: k for k, v in enumerate(sorted(vocab_list))}
        vocab_dict["|"] = vocab_dict[" "]
        del vocab_dict[" "]

        vocab_dict["[UNK]"] = len(vocab_dict)
        vocab_dict["[PAD]"] = len(vocab_dict)

        with open('vocab.json', 'w') as f:
            json.dump(vocab_dict, f)

    @staticmethod
    def _extract_all_chars(batch):
        all_text = " ".join(batch["sentence"])
        vocab = list(set(all_text))
        return {"vocab": [vocab], "all_text": [all_text]}

    def _prepare_dataset(self, batch):
        audio = batch["audio"]

        batch["input_values"] = self.processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
        batch["input_length"] = len(batch["input_values"])

        with self.processor.as_target_processor():
            batch["labels"] = self.processor(batch["sentence"]).input_ids
        return batch

    def train(self):
        model:Wav2Vec2ForCTC = Wav2Vec2ForCTC.from_pretrained(
            "facebook/wav2vec2-base",
            attention_dropout=0.094,
            hidden_dropout=0.047,
            feat_proj_dropout=0.04,
            mask_time_prob=0.082,
            layerdrop=0.041,
            activation_dropout=0.055,
            ctc_loss_reduction="mean",
            pad_token_id=self.processor.tokenizer.pad_token_id,
            vocab_size=len(self.processor.tokenizer),
        )

        model.freeze_feature_extractor()

        training_args = TrainingArguments(
            output_dir='./',
            group_by_length=True,
            per_device_train_batch_size=10,
            gradient_accumulation_steps=2,
            evaluation_strategy="steps",
            num_train_epochs=10,
            gradient_checkpointing=True,
            fp16=True,
            save_steps=500,
            eval_steps=500,
            logging_steps=50,
            learning_rate=5e-5,
            warmup_steps=500,
            save_total_limit=10,
            push_to_hub=True,
            load_best_model_at_end=True,
            greater_is_better=False,
            metric_for_best_model='eval_wer',
        )

        trainer = Trainer(
            model=model,
            data_collator=self.data_collator,
            args=training_args,
            compute_metrics=self._compute_metrics,
            train_dataset=self.train_ds,
            eval_dataset=self.test_ds,
            tokenizer=self.processor.tokenizer,
        )

        trainer.train()

        trainer.push_to_hub()

    def _compute_metrics(self, pred):
        pred_logits = pred.predictions
        pred_ids = np.argmax(pred_logits, axis=-1)

        pred.label_ids[pred.label_ids == -100] = self.processor.tokenizer.pad_token_id

        pred_str = self.processor.batch_decode(pred_ids)

        label_str = self.processor.batch_decode(pred.label_ids, group_tokens=False)

        wer = self.wer_metric.compute(predictions=pred_str, references=label_str)

        return {"wer": wer}
