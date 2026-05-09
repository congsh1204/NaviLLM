#!/usr/bin/env python
import argparse
import sys
from pathlib import Path

from datasets import load_dataset
from peft import LoraConfig
from transformers import TrainingArguments
from trl import SFTTrainer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from vln_choice.qwen_vl.loaders import load_processor, load_vlm
from vln_choice.qwen_vl.sft_format import format_example_for_sft


def main():
    parser = argparse.ArgumentParser(description="LoRA SFT for Qwen-VL choice navigation samples.")
    parser.add_argument("--model_name_or_path", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--train_jsonl", required=True)
    parser.add_argument("--output_dir", default="vln_choice/outputs/checkpoints/qwen_vl_choice_lora")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    args = parser.parse_args()

    processor = load_processor(args.model_name_or_path)
    model = load_vlm(args.model_name_or_path)
    dataset = load_dataset("json", data_files=args.train_jsonl, split="train").map(format_example_for_sft)

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    train_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        logging_steps=10,
        save_steps=500,
        bf16=True,
        remove_unused_columns=False,
    )
    trainer = SFTTrainer(
        model=model,
        args=train_args,
        train_dataset=dataset,
        processing_class=processor,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(args.output_dir)


if __name__ == "__main__":
    main()
