import os
import sys
import json
import base64
import argparse
from io import BytesIO
from typing import List, Dict, Any, Optional, Tuple

import torch
from torch.utils.data import Dataset
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

WEAVE_TOKEN_NAMES = ["<WEAVE_1>", "<WEAVE_2>", "<WEAVE_3>"]
DEFAULT_NUM_WEAVE = 3
STEP_PLACEHOLDERS = ["Step 1.", "Step 2.", "Step 3."]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train WEAVE tokens and WEAVEGen (Reasoning Anchors) from main.py saved data"
    )
    parser.add_argument(
        "--training_data",
        type=str,
        required=True,
        help="Path to vl_training_data_*_new.json (saved by main.py)",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Base VLM path (same as main.py, e.g. Qwen2.5-VL-7B). Default: env MODEL_PATH",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./weavegen_output",
        help="Output directory for WEAVEGen checkpoint",
    )
    parser.add_argument(
        "--num_weave_tokens",
        type=int,
        default=DEFAULT_NUM_WEAVE,
        help=f"Number of <WEAVE> tokens (default {DEFAULT_NUM_WEAVE})",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=2048,
        help="Max sequence length (prompt + WEAVE + steps + answer)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Per-device train batch size",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-5,
        help="Learning rate",
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.1,
        help="Warmup ratio",
    )
    parser.add_argument(
        "--loss_only_on_answer",
        action="store_true",
        default=True,
        help="Only compute loss on optimized_answer tokens (L_util, default True)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device (cuda/cpu)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--full_finetune",
        action="store_true",
        help="Train all parameters (default: LoRA + trainable token embeddings)",
    )
    parser.add_argument(
        "--lora_r",
        type=int,
        default=16,
        help="LoRA rank (default 16)",
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=32,
        help="LoRA alpha (default 32)",
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.05,
        help="LoRA dropout (default 0.05)",
    )
    parser.add_argument(
        "--lora_on_vision",
        action="store_true",
        help="Apply LoRA to vision tower Linear layers as well (default: language-only)",
    )
    parser.add_argument(
        "--merge_lora_before_save",
        action="store_true",
        help="Merge LoRA into base and save one full checkpoint (no separate adapter)",
    )
    args = parser.parse_args()
    if not args.model_path:
        args.model_path = os.environ.get("MODEL_PATH", "").strip() or None
    if not args.model_path:
        raise ValueError("Provide --model_path or set MODEL_PATH")
    return args


def set_seed(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        import numpy as np
        import random
        np.random.seed(seed)
        random.seed(seed)
    except Exception:
        pass


def load_training_data(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        data = [data]
    return data


def decode_image_from_base64(b64: str) -> Image.Image:
    raw = base64.b64decode(b64)
    return Image.open(BytesIO(raw)).convert("RGB")


def get_weave_and_steps_text(
    tokenizer,
    num_weave: int,
    step_placeholders: List[str],
) -> str:
    parts = []
    for i in range(num_weave):
        name = WEAVE_TOKEN_NAMES[i] if i < len(WEAVE_TOKEN_NAMES) else f"<WEAVE_{i+1}>"
        step = step_placeholders[i] if i < len(step_placeholders) else f"Step {i+1}."
        parts.append(f"{name} {step}")
    return " ".join(parts)


class WeaveUtilDataset(Dataset):
    def __init__(
        self,
        data_list: List[Dict],
        tokenizer,
        processor,
        num_weave: int = DEFAULT_NUM_WEAVE,
        max_length: int = 2048,
        step_placeholders: Optional[List[str]] = None,
    ):
        self.data_list = data_list
        self.tokenizer = tokenizer
        self.processor = processor
        self.num_weave = num_weave
        self.max_length = max_length
        self.step_placeholders = step_placeholders or STEP_PLACEHOLDERS
        self.weave_steps_suffix = get_weave_and_steps_text(
            tokenizer, num_weave, self.step_placeholders
        )

    def __len__(self) -> int:
        return len(self.data_list)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data_list[idx]
        prompt = item["prompt"]
        image_b64 = item["image_base64"]
        optimized_answer = item.get("optimized_answer", "").strip()
        if not optimized_answer:
            optimized_answer = item.get("ground_truth", "")

        image = decode_image_from_base64(image_b64)

        prompt_with_weave = prompt + "\n\n" + self.weave_steps_suffix + "\n\n"

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt_with_weave + optimized_answer},
                ],
            }
        ]

        try:
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
        except Exception:
            text = prompt_with_weave + optimized_answer

        try:
            inputs = self.processor(
                text=[text],
                images=[image],
                padding="max_length",
                max_length=self.max_length,
                truncation=True,
                return_tensors="pt",
            )
        except Exception:
            inputs = self.processor(
                messages=messages,
                padding="max_length",
                max_length=self.max_length,
                truncation=True,
                return_tensors="pt",
            )

        def _squeeze(x):
            if isinstance(x, torch.Tensor) and x.dim() > 0:
                return x.squeeze(0)
            return x

        input_ids = _squeeze(inputs["input_ids"])
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = _squeeze(attention_mask)
        else:
            pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
            attention_mask = (input_ids != pad_id).long()
            if attention_mask.sum() == 0:
                attention_mask = torch.ones_like(input_ids)

        answer_ids = self.tokenizer(
            optimized_answer,
            add_special_tokens=False,
            return_tensors="pt",
        )["input_ids"].squeeze(0)
        L_answer = min(answer_ids.shape[0], input_ids.shape[0])
        labels = input_ids.clone()
        labels[:-L_answer] = -100
        if self.tokenizer.pad_token_id is not None:
            labels[labels == self.tokenizer.pad_token_id] = -100

        out = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
        for k in ("pixel_values", "image_grid_thw", "position_ids", "video_pixel_values"):
            if k in inputs and inputs[k] is not None:
                out[k] = _squeeze(inputs[k])
        return out


def add_weave_tokens_to_tokenizer(tokenizer, num_weave: int) -> List[str]:
    to_add = []
    for i in range(num_weave):
        name = WEAVE_TOKEN_NAMES[i] if i < len(WEAVE_TOKEN_NAMES) else f"<WEAVE_{i+1}>"
        if name not in tokenizer.get_vocab():
            to_add.append(name)
    if to_add:
        tokenizer.add_special_tokens({"additional_special_tokens": to_add})
    return to_add


def _module_name_for_submodule(root: torch.nn.Module, target: torch.nn.Module) -> Optional[str]:
    for name, mod in root.named_modules():
        if mod is target:
            return name
    return None


def collect_lora_target_suffixes(model: torch.nn.Module, include_vision: bool) -> List[str]:
    allowed = {
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    }
    found: set = set()
    for name, mod in model.named_modules():
        if not isinstance(mod, torch.nn.Linear):
            continue
        short = name.rsplit(".", 1)[-1]
        if short not in allowed:
            continue
        if not include_vision:
            low = name.lower()
            if "visual" in low or "vision_tower" in low:
                continue
        found.add(short)
    if not found:
        return sorted(allowed)
    return sorted(found)


def build_modules_to_save(model: torch.nn.Module) -> List[str]:
    names: List[str] = []
    emb = model.get_input_embeddings()
    emb_name = _module_name_for_submodule(model, emb)
    if emb_name:
        names.append(emb_name)
    tied = getattr(model.config, "tie_word_embeddings", True)
    if not tied and hasattr(model, "lm_head"):
        head_name = _module_name_for_submodule(model, model.lm_head)
        if head_name:
            names.append(head_name)
    return names or ["model.embed_tokens"]


def apply_lora_weavegen(model: torch.nn.Module, args: argparse.Namespace) -> torch.nn.Module:
    try:
        from peft import LoraConfig, TaskType, get_peft_model
    except ImportError as e:
        raise ImportError(
            "LoRA 训练需要安装 peft：pip install peft>=0.10.0"
        ) from e

    target_modules = collect_lora_target_suffixes(model, include_vision=args.lora_on_vision)
    modules_to_save = build_modules_to_save(model)
    print(f"LoRA target_modules (suffix match): {target_modules}")
    print(f"modules_to_save (new tokens / embeddings): {modules_to_save}")

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias="none",
        modules_to_save=modules_to_save,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def main():
    args = parse_args()
    set_seed(args.seed)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print(f"Loading base model from {args.model_path} ...")
    try:
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        model_cls = Qwen2_5_VLForConditionalGeneration
    except ImportError:
        from transformers import Qwen2VLForConditionalGeneration as model_cls
    model = model_cls.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
    )
    try:
        processor = AutoProcessor.from_pretrained(args.model_path)
    except Exception:
        from transformers import Qwen2VLProcessor
        processor = Qwen2VLProcessor.from_pretrained(args.model_path)

    tokenizer = processor.tokenizer
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    num_added = add_weave_tokens_to_tokenizer(tokenizer, args.num_weave_tokens)
    print(f"Added WEAVE tokens: {num_added}")
    model.resize_token_embeddings(len(tokenizer))
    processor.tokenizer = tokenizer

    use_lora = not args.full_finetune
    if use_lora:
        model = apply_lora_weavegen(model, args)
    else:
        print("Full fine-tuning: all parameters trainable.")

    data_list = load_training_data(args.training_data)
    print(f"Loaded {len(data_list)} samples from {args.training_data}")

    dataset = WeaveUtilDataset(
        data_list=data_list,
        tokenizer=tokenizer,
        processor=processor,
        num_weave=args.num_weave_tokens,
        max_length=args.max_length,
    )

    from torch.utils.data import DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn_with_padding,
        num_workers=0,
    )

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr)
    total_steps = len(dataloader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        return max(0.0, (total_steps - step) / (total_steps - warmup_steps))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    os.makedirs(args.output_dir, exist_ok=True)

    def _model_device():
        if hasattr(model, "device"):
            return model.device
        return next(model.parameters()).device

    global_step = 0
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch in pbar:
            optimizer.zero_grad()
            dev = _model_device()
            batch = {k: v.to(dev) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            labels = batch.pop("labels", None)
            if labels is not None:
                batch["labels"] = labels

            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            scheduler.step()
            global_step += 1
            epoch_loss += loss.item()
            pbar.set_postfix(loss=loss.item())
        print(f"Epoch {epoch+1} avg loss: {epoch_loss / len(dataloader):.4f}")

    if use_lora and args.merge_lora_before_save:
        print("Merging LoRA into base weights before save ...")
        model = model.merge_and_unload()
        model.save_pretrained(args.output_dir)
    else:
        model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    if hasattr(processor, "save_pretrained"):
        processor.save_pretrained(args.output_dir)

    if use_lora and not args.merge_lora_before_save:
        meta = {
            "base_model_path": os.path.abspath(args.model_path),
            "finetune": "lora",
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "lora_on_vision": args.lora_on_vision,
        }
        meta_path = os.path.join(args.output_dir, "weave_lora_meta.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        print(
            f"Saved LoRA adapter to {args.output_dir}. "
            f"Evaluate with the same base model path (see weave_lora_meta.json or eval_weave.py --base_model_path)."
        )
    print(f"WEAVEGen saved to {args.output_dir}")


def collate_fn_with_padding(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    keys = batch[0].keys()
    out = {}
    for k in keys:
        vals = [b[k] for b in batch]
        if isinstance(vals[0], torch.Tensor):
            if vals[0].dim() == 1:
                pad_val = -100 if k == "labels" else 0
                out[k] = torch.nn.utils.rnn.pad_sequence(
                    vals, batch_first=True, padding_value=pad_val
                )
            else:
                out[k] = torch.stack(vals, dim=0)
        else:
            out[k] = vals
    return out


if __name__ == "__main__":
    main()
