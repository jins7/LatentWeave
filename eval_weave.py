import os
import sys
import argparse
import json
from datetime import datetime
from typing import List, Dict, Any, Optional

import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_loader import load_mmmu_dataset, load_mathvision_dataset, format_question_with_options
from prompts import get_mmmu_solver_prompt
from utils import is_answer_correct, extract_answer_from_text, mask_path

WEAVE_TOKEN_NAMES = ["<WEAVE_1>", "<WEAVE_2>", "<WEAVE_3>"]
STEP_PLACEHOLDERS = ["Step 1.", "Step 2.", "Step 3."]
DEFAULT_NUM_WEAVE = 3


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate WEAVEGen with <WEAVE> tokens on multimodal reasoning datasets"
    )
    parser.add_argument(
        "--weavegen_path",
        type=str,
        required=True,
        help="Path to WEAVEGen checkpoint (output_dir of train_weave_anchors.py)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="mmmu",
        choices=["mmmu", "mathvision"],
        help="Dataset to evaluate on (default: mmmu)",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Cache directory for HuggingFace datasets",
    )
    parser.add_argument(
        "--subsets",
        type=str,
        nargs="+",
        default=None,
        help="MMMU subset names (default: all); ignored for other datasets",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./eval_outputs",
        help="Directory for result jsonl and summary",
    )
    parser.add_argument(
        "--start_idx",
        type=int,
        default=0,
        help="Start index of samples (default: 0)",
    )
    parser.add_argument(
        "--end_idx",
        type=int,
        default=None,
        help="End index of samples (None = all)",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=1024,
        help="Max new tokens for generation (default: 1024)",
    )
    parser.add_argument(
        "--num_weave_tokens",
        type=int,
        default=DEFAULT_NUM_WEAVE,
        help=f"Number of WEAVE tokens in prompt (default {DEFAULT_NUM_WEAVE})",
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
        "--verbose",
        action="store_true",
        help="Print each sample prompt and answer",
    )
    args = parser.parse_args()
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


def get_weave_suffix(num_weave: int) -> str:
    parts = []
    for i in range(num_weave):
        name = WEAVE_TOKEN_NAMES[i] if i < len(WEAVE_TOKEN_NAMES) else f"<WEAVE_{i+1}>"
        step = STEP_PLACEHOLDERS[i] if i < len(STEP_PLACEHOLDERS) else f"Step {i+1}."
        parts.append(f"{name} {step}")
    return " ".join(parts)


def weave_generate(
    image: Image.Image,
    question: str,
    model,
    processor,
    device: str,
    max_new_tokens: int = 1024,
    num_weave: int = DEFAULT_NUM_WEAVE,
) -> str:
    weave_suffix = get_weave_suffix(num_weave)
    full_text = question + "\n\n" + weave_suffix + "\n\n"

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": full_text},
            ],
        }
    ]

    try:
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        text = full_text

    try:
        inputs = processor(
            text=[text],
            images=[image],
            padding=True,
            return_tensors="pt",
        )
    except Exception:
        inputs = processor(
            messages=messages,
            padding=True,
            return_tensors="pt",
        )

    if hasattr(inputs, "to"):
        inputs = inputs.to(device)
    else:
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    input_ids = inputs["input_ids"]
    if input_ids is None:
        raise RuntimeError("processor did not return input_ids")
    input_length = input_ids.shape[1]
    pad_id = processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=pad_id,
        )

    generated_ids = output[0][input_length:]
    answer_text = processor.tokenizer.decode(generated_ids, skip_special_tokens=True)
    return answer_text.strip()


def run_eval(
    model,
    processor,
    all_data: List[Dict],
    start_idx: int,
    end_idx: int,
    device: str,
    args,
) -> tuple:
    results = []
    correct = 0
    total = 0
    pbar = tqdm(range(start_idx, end_idx), desc="WEAVEGen eval", unit="sample")

    for idx in pbar:
        sample = all_data[idx]
        image = sample.get("image_1")
        if image is None:
            continue
        if not isinstance(image, Image.Image):
            try:
                image = Image.open(image).convert("RGB")
            except Exception:
                continue

        question = sample.get("question", "")
        options = sample.get("options", [])
        if not question:
            continue

        question_with_options = format_question_with_options(question, options)
        subset_name = sample.get("subset", "")
        ground_truth = sample.get("answer", "")
        has_options = len(options) > 0

        messages = get_mmmu_solver_prompt(question_with_options, has_options=has_options)
        if isinstance(messages, list) and len(messages) > 0:
            formatted_question = messages[-1].get("content", question_with_options)
        else:
            formatted_question = question_with_options

        if args.verbose:
            print(f"\n[{idx}] {subset_name}")
            print("Prompt (with WEAVE):", formatted_question[:200], "...")

        try:
            answer_text = weave_generate(
                image=image,
                question=formatted_question,
                model=model,
                processor=processor,
                device=device,
                max_new_tokens=args.max_new_tokens,
                num_weave=args.num_weave_tokens,
            )
        except Exception as e:
            answer_text = ""
            if args.verbose:
                print(f"Generate error: {e}")

        extracted = extract_answer_from_text(answer_text, has_options=has_options)
        is_correct = is_answer_correct(answer_text, ground_truth, has_options=has_options)
        if is_correct:
            correct += 1
        total += 1

        rec = {
            "subset": subset_name,
            "question": question[:200],
            "weave_answer": answer_text,
            "extracted_answer": extracted,
            "ground_truth": ground_truth,
            "correct": is_correct,
        }
        results.append(rec)
        pbar.set_postfix(correct=correct, total=total, acc=f"{correct/max(1,total):.3f}")

        if args.verbose:
            print("Answer:", answer_text[:300])
            print("Extracted:", extracted, "GT:", ground_truth, "Correct:", is_correct)

    return results, correct, total


def main():
    args = parse_args()
    set_seed(args.seed)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print(f"Loading WEAVEGen from {mask_path(args.weavegen_path)} ...")
    try:
        from transformers import Qwen2_5_VLForConditionalGeneration
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.weavegen_path,
            torch_dtype=torch.bfloat16,
            device_map="auto" if device == "cuda" else None,
            trust_remote_code=True,
        )
        print("Loaded Qwen2_5_VLForConditionalGeneration (WEAVEGen)")
    except Exception:
        try:
            from transformers import Qwen2VLForConditionalGeneration
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                args.weavegen_path,
                torch_dtype=torch.bfloat16,
                device_map="auto" if device == "cuda" else None,
                trust_remote_code=True,
            )
            print("Loaded Qwen2VLForConditionalGeneration (WEAVEGen)")
        except Exception as e:
            raise RuntimeError(f"Failed to load WEAVEGen from {mask_path(args.weavegen_path)}: {e}") from e

    try:
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(args.weavegen_path)
    except Exception:
        try:
            from transformers import Qwen2VLProcessor
            processor = Qwen2VLProcessor.from_pretrained(args.weavegen_path)
        except Exception:
            from transformers import Qwen2_5_VLProcessor
            processor = Qwen2_5_VLProcessor.from_pretrained(args.weavegen_path)

    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id

    model.eval()
    if args.dataset == "mathvision":
        all_data = load_mathvision_dataset(cache_dir=args.cache_dir, split="testmini")
        dataset_prefix = "mathvision"
    else:
        all_data = load_mmmu_dataset(cache_dir=args.cache_dir, subsets=args.subsets)
        dataset_prefix = "mmmu"
    if not all_data:
        print("No data loaded.")
        return
    print(f"Loaded {len(all_data)} samples ({args.dataset})")
    start_idx = max(0, args.start_idx)
    end_idx = args.end_idx if args.end_idx is not None else len(all_data)
    end_idx = min(end_idx, len(all_data))

    print(f"Eval range: [{start_idx}, {end_idx})")

    results, correct, total = run_eval(
        model, processor, all_data, start_idx, end_idx, device, args
    )

    accuracy = correct / total if total > 0 else 0.0
    print(f"\n{'='*60}")
    print("WEAVEGen Evaluation Summary")
    print(f"{'='*60}")
    print(f"Dataset: {args.dataset}")
    print(f"Total: {total}, Correct: {correct}, Accuracy: {accuracy:.4f}")
    print(f"{'='*60}\n")

    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_jsonl = os.path.join(
        args.output_dir,
        f"{dataset_prefix}_weavegen_eval_{timestamp}.jsonl",
    )
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Results saved to {mask_path(out_jsonl)}")

    summary_path = os.path.join(
        args.output_dir,
        f"{dataset_prefix}_weavegen_summary_{timestamp}.json",
    )
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset": args.dataset,
                "weavegen_path": args.weavegen_path,
                "total": total,
                "correct": correct,
                "accuracy": accuracy,
                "start_idx": start_idx,
                "end_idx": end_idx,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"Summary saved to {mask_path(summary_path)}")


if __name__ == "__main__":
    main()
