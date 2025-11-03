import os

from datasets import load_dataset, Audio
from transformers import WhisperProcessor, WhisperForConditionalGeneration, Seq2SeqTrainer, Seq2SeqTrainingArguments
# import torch

MODEL_NAME = "openai/whisper-small"
MANIFEST_FILE = "data/processed/manifest.jsonl"

print("Loading dataset...")
dataset = load_dataset("json", data_files=MANIFEST_FILE, split="train")
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

print("getting processor and model")
print("downloading processor")
processor = WhisperProcessor.from_pretrained(MODEL_NAME, token=os.environ.get("HF_TOKEN"))
print("downloading model")
model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME, token=os.environ.get("HF_TOKEN")).to("cuda")
print("downloading completed")


def prepare(example):
    # Process audio
    audio_inputs = processor.feature_extractor(
        example["audio"]["array"],
        sampling_rate=16000,
        return_tensors="pt"
    )

    # Tokenize transcript with padding
    transcript_inputs = processor.tokenizer(
        example["text"],
        padding="max_length",        # or "longest"
        truncation=True,
        max_length=448,              # safe length for Whisper small
        return_tensors="pt"
    )

    return {
        "input_features": audio_inputs.input_features[0],
        "labels": transcript_inputs.input_ids[0],
    }

print("Preparing dataset...")
dataset = dataset.map(prepare, remove_columns=dataset.column_names)

print("Starting training...")
training_args = Seq2SeqTrainingArguments(
    output_dir="./models/whisper-small",
    gradient_accumulation_steps=2,
    learning_rate=1e-5,
    warmup_steps=500,
    fp16=False,
    logging_dir="./logs",
    logging_strategy="steps",
    logging_steps=10,
    save_steps=500,
    eval_steps=500,
    save_total_limit=2,
    report_to="none",
    per_device_train_batch_size=1,
    max_steps=500

)

print("Starting training...")
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    processing_class=processor.feature_extractor,
)

trainer.train()