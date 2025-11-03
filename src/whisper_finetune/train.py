"""Fine-tune a Whisper checkpoint on a manifest built by `prepare`."""

from __future__ import annotations

from functools import partial

from transformers import Seq2SeqTrainingArguments

from .config import TrainConfig


def build_training_arguments(config: TrainConfig, device: str) -> Seq2SeqTrainingArguments:
    """Build the trainer arguments without touching the model or the dataset."""
    evaluate = config.eval_fraction > 0
    return Seq2SeqTrainingArguments(
        output_dir=str(config.output_dir),
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_steps=config.warmup_steps,
        max_steps=config.max_steps,
        # Mixed precision only where it is supported and pays off.
        fp16=device == "cuda",
        use_cpu=device == "cpu",
        logging_strategy="steps",
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        eval_strategy="steps" if evaluate else "no",
        eval_steps=config.save_steps if evaluate else None,
        seed=config.seed,
        report_to="none",
    )


def run_training(config: TrainConfig, device: str) -> None:
    """Load the processor, model and manifest, then train and save to ``output_dir``.

    Weights come from the Hugging Face Hub (or its local cache); a token for
    gated or private checkpoints is read from ``HF_TOKEN`` by huggingface_hub
    itself, so it never passes through this code.
    """
    from datasets import load_dataset
    from transformers import Seq2SeqTrainer, WhisperForConditionalGeneration, WhisperProcessor

    from .features import SpeechSeq2SeqCollator, prepare_example

    args = build_training_arguments(config, device)
    processor = WhisperProcessor.from_pretrained(config.model_name, language=config.language, task=config.task)
    model = WhisperForConditionalGeneration.from_pretrained(config.model_name)
    model.generation_config.language = config.language
    model.generation_config.task = config.task
    model.generation_config.forced_decoder_ids = None

    dataset = load_dataset("json", data_files=str(config.manifest), split="train")
    dataset = dataset.map(
        partial(
            prepare_example,
            feature_extractor=processor.feature_extractor,
            tokenizer=processor.tokenizer,
            max_label_length=config.max_label_length,
        ),
        remove_columns=dataset.column_names,
    )
    eval_dataset = None
    if config.eval_fraction > 0:
        split = dataset.train_test_split(test_size=config.eval_fraction, seed=config.seed)
        dataset, eval_dataset = split["train"], split["test"]

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        data_collator=SpeechSeq2SeqCollator(model.config.decoder_start_token_id),
        processing_class=processor,
    )
    trainer.train()
    trainer.save_model()
    processor.save_pretrained(str(config.output_dir))
