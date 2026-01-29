from typing import List, Dict, Any, Optional
from PIL import Image
from datasets import load_dataset, get_dataset_config_names

MMMU_DATASET_NAME = "MMMU/MMMU"
MATHVISION_DATASET_NAME = "MathLLMs/MathVision"


def load_mmmu_dataset(
    cache_dir: Optional[str] = None,
    subsets: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    if subsets is None:
        try:
            print(f"Fetching {MMMU_DATASET_NAME} subset configs...")
            subsets = get_dataset_config_names(MMMU_DATASET_NAME, cache_dir=cache_dir)
            print(f"Found {len(subsets)} subsets")
        except Exception as e:
            raise RuntimeError(f"Failed to read {MMMU_DATASET_NAME} config: {e}")
    all_data = []
    print(f"Loading MMMU, {len(subsets)} subsets...")
    for subset_idx, subset_name in enumerate(subsets, 1):
        try:
            print(f"[{subset_idx}/{len(subsets)}] {subset_name}...")
            dataset = load_dataset(
                "MMMU/MMMU",
                subset_name,
                split="validation",
                cache_dir=cache_dir
            )
            for example in dataset:
                question = example.get("question", "")
                options = example.get("options", [])
                image_1 = example.get("image_1", None)
                answer = example.get("answer", "")
                if image_1 is not None:
                    if isinstance(image_1, Image.Image):
                        image = image_1
                    elif isinstance(image_1, str):
                        try:
                            image = Image.open(image_1).convert('RGB')
                        except Exception as e:
                            print(f"Warning: cannot load image {image_1}: {e}")
                            continue
                    elif hasattr(image_1, 'convert'):
                        image = image_1.convert('RGB')
                    else:
                        print(f"Warning: unknown image type {type(image_1)}")
                        continue
                else:
                    print("Warning: missing image_1")
                    continue
                if not question:
                    print("Warning: missing question")
                    continue
                if isinstance(options, dict):
                    ordered_keys = ["A", "B", "C", "D", "a", "b", "c", "d", "0", "1", "2", "3"]
                    tmp = []
                    for k in ordered_keys:
                        if k in options and options[k] not in tmp:
                            tmp.append(str(options[k]))
                    if not tmp:
                        tmp = [str(v) for v in options.values()]
                    options = tmp
                elif isinstance(options, str):
                    import ast
                    try:
                        parsed = ast.literal_eval(options)
                        if isinstance(parsed, list):
                            options = [str(item) for item in parsed]
                        else:
                            options = [str(parsed)]
                    except (ValueError, SyntaxError):
                        options = [options]
                elif not isinstance(options, list):
                    try:
                        options = [str(item) for item in list(options)]
                    except Exception:
                        options = [str(options)]
                options = [str(opt).strip() for opt in options if opt]
                if answer is None:
                    answer = ""
                else:
                    answer = str(answer).strip()
                data_item = {
                    "subset": subset_name,
                    "question": question,
                    "options": options,
                    "image_1": image,
                    "answer": answer.upper() if len(answer) == 1 else answer,
                }
                all_data.append(data_item)
            print(f"  {subset_name}: {len(dataset)} samples")
        except Exception as e:
            print(f"  Failed {subset_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    print(f"\nLoaded {len(all_data)} samples")
    return all_data


def format_question_with_options(question: str, options: List[str]) -> str:
    formatted = f"Question:\n{question}\n"
    if options:
        formatted += "\nOptions:\n"
        labels = [chr(ord("A") + idx) for idx in range(len(options))]
        for label, option in zip(labels, options):
            formatted += f"{label}. {option}\n"
    return formatted


def load_mathvision_dataset(
    cache_dir: Optional[str] = None,
    split: str = "testmini"
) -> List[Dict[str, Any]]:
    all_data = []
    print(f"Loading MathVision split {split}...")
    try:
        print(f"Loading {MATHVISION_DATASET_NAME} (split: {split})...")
        dataset = load_dataset(
            MATHVISION_DATASET_NAME,
            split=split,
            cache_dir=cache_dir
        )
        split_data = dataset
        total_count = 0
        mcq_count = 0
        open_ended_count = 0
        for example in split_data:
            total_count += 1
            question = example.get("question", "")
            options = example.get("options", [])
            decoded_image = example.get("decoded_image", None)
            answer = example.get("answer", "")
            if not isinstance(options, list):
                if isinstance(options, dict):
                    ordered_keys = ["A", "B", "C", "D", "a", "b", "c", "d", "0", "1", "2", "3"]
                    tmp = []
                    for k in ordered_keys:
                        if k in options and options[k] not in tmp:
                            tmp.append(str(options[k]))
                    if not tmp:
                        tmp = [str(v) for v in options.values()]
                    options = tmp
                elif isinstance(options, str):
                    import ast
                    try:
                        parsed = ast.literal_eval(options)
                        if isinstance(parsed, list):
                            options = [str(item) for item in parsed]
                        else:
                            options = [str(parsed)]
                    except (ValueError, SyntaxError):
                        options = [options]
                else:
                    try:
                        options = [str(item) for item in list(options)]
                    except Exception:
                        options = []
            options = [str(opt).strip() for opt in options if opt]
            if len(options) > 0:
                mcq_count += 1
            else:
                open_ended_count += 1
            if decoded_image is not None:
                if isinstance(decoded_image, Image.Image):
                    image = decoded_image
                elif isinstance(decoded_image, str):
                    try:
                        image = Image.open(decoded_image).convert('RGB')
                    except Exception as e:
                        print(f"Warning: cannot load {decoded_image}: {e}")
                        continue
                elif hasattr(decoded_image, 'convert'):
                    image = decoded_image.convert('RGB')
                else:
                    print(f"Warning: unknown image type {type(decoded_image)}")
                    continue
            else:
                print("Warning: missing decoded_image")
                continue
            if not question:
                print("Warning: missing question")
                continue
            if answer is None:
                answer = ""
            else:
                answer = str(answer).strip()
            if len(options) > 0:
                if len(answer) == 1:
                    answer = answer.upper()
                else:
                    answer_lower = answer.lower()
                    for idx, opt in enumerate(options):
                        if opt.lower() == answer_lower or answer_lower in opt.lower() or opt.lower() in answer_lower:
                            answer = chr(ord("A") + idx)
                            break
            data_item = {
                "subset": "MathVision",
                "question": question,
                "options": options,
                "image_1": image,
                "answer": answer,
            }
            all_data.append(data_item)
        print(f"  {split}: {total_count} (MCQ {mcq_count}, open {open_ended_count})")
    except Exception as e:
        print(f"  MathVision load failed: {e}")
        import traceback
        traceback.print_exc()
        raise
    print(f"\nMathVision: {len(all_data)} (MCQ {mcq_count}, open {open_ended_count})")
    return all_data
