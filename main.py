import os
import sys
import torch
import argparse
from tqdm import tqdm
import json
from datetime import datetime
from PIL import Image


import re


class Tee:
    """Duplicate stdout/stderr to console and log file; strip ANSI from log."""
    ANSI_ESCAPE = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')

    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for i, f in enumerate(self.files):
            if i == 0:
                f.write(obj)
            else:
                cleaned_obj = self.ANSI_ESCAPE.sub('', obj)
                f.write(cleaned_obj)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()



os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("CUDNN_DETERMINISTIC", "1")
sys.path.append(os.path.dirname(__file__))

from data_loader import load_mmmu_dataset, load_mathvision_dataset, format_question_with_options
from prompts import get_mmmu_solver_prompt
from vl_generation import original_vl_generation, extract_latent_segments
from vl_optimization import optimize_all_segments
from utils import is_answer_correct, extract_answer_from_text, mask_path
from reward_model import VLRewardModel
from collect_pre_segment3_data import collect_pre_segment3_from_regenerated, save_training_data

training_data_list = []


def parse_args():
    parser = argparse.ArgumentParser(description="Process MMMU or MathVision dataset with latent optimization")
    env_model_path = (os.environ.get("MODEL_PATH") or "").strip()
    env_output_dir = (os.environ.get("OUTPUT_DIR") or "").strip()
    default_model_path = env_model_path or ""
    default_output_dir = env_output_dir or "./outputs"

    parser.add_argument("--model_path", type=str, default=default_model_path, help="Path to Qwen2-VL model (or set MODEL_PATH)")
    parser.add_argument("--cache_dir", type=str, default=None, help="Cache directory for HuggingFace datasets")
    parser.add_argument("--subsets", type=str, nargs="+", default=None, help="MMMU subset names (default: all)")
    parser.add_argument("--dataset", type=str, default="mmmu", choices=["mmmu", "mathvision"], help="Dataset: mmmu or mathvision")
    parser.add_argument("--output_dir", type=str, default=default_output_dir, help="Output directory (or set OUTPUT_DIR)")

    parser.add_argument("--max_num_steps", type=int, default=5, help="Optimization steps per segment")
    parser.add_argument("--lr", type=float, default=0.03, help="Learning rate")
    parser.add_argument("--grad_clip", type=float, default=None, help="Gradient clipping threshold")
    parser.add_argument("--reward_threshold", type=float, default=-0.2, help="Reward threshold to stop")
    parser.add_argument("--segment_length", type=int, default=20, help="Latent segment length")
    parser.add_argument("--max_new_tokens", type=int, default=1024, help="Max new tokens to generate")

    parser.add_argument("--fast_mode", action="store_true", help="Fast mode: max_num_steps=5, max_new_tokens=512, segment_length=15")
    parser.add_argument("--ultra_fast_mode", action="store_true", help="Ultra fast: max_num_steps=3, max_new_tokens=256, segment_length=10")

    parser.add_argument("--start_idx", type=int, default=0, help="Start index")
    parser.add_argument("--end_idx", type=int, default=None, help="End index (None = all)")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--disable_cudnn", action="store_true", help="Disable cuDNN")
    parser.add_argument("--log_file", type=str, default=None, help="Log file path")
    parser.add_argument("--ground_truth_weight", type=float, default=5.0, help="Ground-truth reward weight (0 to disable)")

    args = parser.parse_args()
    if not (args.model_path or "").strip():
        args.model_path = default_model_path
    if not (args.output_dir or "").strip():
        args.output_dir = default_output_dir
    if not (args.model_path or "").strip():
        raise ValueError("Set MODEL_PATH or provide --model_path")
    return args


def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        try:
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except Exception as e:
            print(f"Warning: CUDA seed failed: {e}, continuing with CPU.")
    import numpy as np
    import random
    np.random.seed(seed)
    random.seed(seed)


def process_single_sample(sample: dict, model, processor, reward_model, device: str, args) -> dict:
    try:
        image = sample.get('image_1')
        if image is None:
            print("Warning: sample missing image_1")
            return None
        if not isinstance(image, Image.Image):
            try:
                image = Image.open(image).convert('RGB')
            except Exception as e:
                print(f"Warning: failed to load image: {e}")
                return None
        question = sample.get('question', '')
        options = sample.get('options', [])
        
        if not question:
            print("Warning: sample missing question")
            return None
        question_with_options = format_question_with_options(question, options)
        has_options = len(options) > 0
        messages = get_mmmu_solver_prompt(question_with_options, has_options=has_options)
        if isinstance(messages, list) and len(messages) > 0:
            formatted_question = messages[-1].get('content', question_with_options)
        else:
            formatted_question = question_with_options
        print(f"\n{'='*80}")
        print("Full prompt:")
        print(f"{'='*80}")
        if isinstance(messages, list):
            for msg in messages:
                role = msg.get('role', 'unknown')
                content = msg.get('content', '')
                print(f"\n[{role.upper()}]")
                print(content)
        else:
            print(formatted_question)
        print(f"{'='*80}\n")
        ground_truth = sample.get('answer', '')
        original_answer, hidden_states_list, input_ids, generated_token_ids = original_vl_generation(
            image=image,
            question=formatted_question,
            model=model,
            processor=processor,
            device=device,
            max_new_tokens=args.max_new_tokens,
            dataset=args.dataset
        )
        original_reward = reward_model.get_reward(question_with_options, original_answer, image=image, processor=processor, ground_truth=ground_truth, has_options=has_options)
        segments = extract_latent_segments(
            hidden_states_list=hidden_states_list,
            generated_text=original_answer,
            token_ids=generated_token_ids,
            tokenizer=processor.tokenizer,
            segment_length=args.segment_length,
            use_percentage=True,
            percentage=0.05
        )
        if not segments:
            ground_truth = sample.get('answer', '')
            original_is_correct = is_answer_correct(original_answer, ground_truth, has_options=has_options)
            original_extracted_answer = extract_answer_from_text(original_answer, has_options=has_options)
            
            return {
                'subset': sample.get('subset', 'unknown'),
                'question': question,
                'options': options,
                'original_answer': original_answer,
                'optimized_answer': original_answer,
                'original_extracted_answer': original_extracted_answer,
                'optimized_extracted_answer': original_extracted_answer,
                'original_reward': float(original_reward),
                'optimized_reward': float(original_reward),
                'num_segments': 0,
                'reward_histories': [],
                'ground_truth_answer': ground_truth,
                'original_is_correct': original_is_correct,
                'optimized_is_correct': original_is_correct
            }
        base_input_ids = input_ids[:, :input_ids.shape[1] - len(hidden_states_list)]
        if len(segments) > 0:
            optimized_answer, reward_histories, optimized_length, segment_results, second_all_hidden_states = optimize_all_segments(
                segments=segments,
                reward_model=reward_model,
                model=model,
                processor=processor,
                device=device,
                question=question_with_options,
                original_answer=original_answer,
                original_input_ids=input_ids,
                base_input_ids=base_input_ids,
                image=image,
                max_num_steps=args.max_num_steps,
                lr=args.lr,
                grad_clip=args.grad_clip,
                reward_threshold=args.reward_threshold,
                max_new_tokens=args.max_new_tokens,
                ground_truth=ground_truth,
                has_options=has_options,
                dataset=args.dataset
            )
        else:
            optimized_answer = original_answer
            reward_histories = []
            optimized_length = len(processor.tokenizer.encode(original_answer))
            segment_results = []
            second_all_hidden_states = None
        optimized_reward = reward_model.get_reward(question_with_options, optimized_answer, image=image, processor=processor, ground_truth=ground_truth, has_options=has_options)
        original_is_correct = is_answer_correct(original_answer, ground_truth, has_options=has_options)
        original_extracted_answer = extract_answer_from_text(original_answer, has_options=has_options)
        optimized_is_correct = is_answer_correct(optimized_answer, ground_truth, has_options=has_options)
        optimized_extracted_answer = extract_answer_from_text(optimized_answer, has_options=has_options)
        if optimized_is_correct and len(segment_results) >= 3 and second_all_hidden_states is not None:
            training_data = collect_pre_segment3_from_regenerated(
                sample=sample,
                original_hidden_states_list=hidden_states_list,
                original_segments=segments,
                segment_results=segment_results,
                image=image,
                question_with_options=question_with_options,
                optimized_is_correct=optimized_is_correct,
                second_regenerated_hidden_states=second_all_hidden_states,
                original_answer=original_answer,
                optimized_answer=optimized_answer
            )
            if training_data is not None:
                training_data_list.append(training_data)
                print(f"Collected training data, total: {len(training_data_list)}")
        
        result = {
            'subset': sample.get('subset', 'unknown'),
            'question': question,
            'options': options,
            'original_answer': original_answer,
            'optimized_answer': optimized_answer,
            'original_extracted_answer': original_extracted_answer,
            'optimized_extracted_answer': optimized_extracted_answer,
            'original_reward': float(original_reward),
            'optimized_reward': float(optimized_reward),
            'num_segments': len(segments),
            'reward_histories': [[float(r) for r in rh] for rh in reward_histories],
            'ground_truth_answer': ground_truth,
            'original_is_correct': original_is_correct,
            'optimized_is_correct': optimized_is_correct
        }
        
        return result
    
    except Exception as e:
        print(f"Error processing sample: {e}")
        import traceback
        traceback.print_exc()
        return None


def main(args):
    if args.log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(args.output_dir, f"log_{timestamp}.txt")
    else:
        log_file = args.log_file
    os.makedirs(args.output_dir, exist_ok=True)
    os.environ['TERM'] = os.environ.get('TERM', 'xterm-256color')
    os.environ['FORCE_COLOR'] = '1'
    log_f = open(log_file, 'w', encoding='utf-8')
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = Tee(original_stdout, log_f)
    sys.stderr = Tee(original_stderr, log_f)
    
    try:
        if args.disable_cudnn:
            print("Warning: cuDNN disabled, performance may drop")
            torch.backends.cudnn.enabled = False
            os.environ["CUDNN_DISABLE"] = "1"
        if args.seed:
            set_seed(args.seed)
        if args.ultra_fast_mode:
            print("=" * 60)
            print("Ultra Fast Mode")
            print("=" * 60)
            args.max_num_steps = 3
            args.max_new_tokens = 256
            args.segment_length = 10
            args.reward_threshold = 0.1
            print(f"  max_num_steps: {args.max_num_steps}")
            print(f"  max_new_tokens: {args.max_new_tokens}")
            print(f"  segment_length: {args.segment_length}")
            print(f"  reward_threshold: {args.reward_threshold}")
            print("=" * 60)
        elif args.fast_mode:
            print("=" * 60)
            print("Fast Mode")
            print("=" * 60)
            args.max_num_steps = 5
            args.max_new_tokens = 512
            args.segment_length = 15
            args.reward_threshold = 0.0
            print(f"  max_num_steps: {args.max_num_steps}")
            print(f"  max_new_tokens: {args.max_new_tokens}")
            print(f"  segment_length: {args.segment_length}")
            print(f"  reward_threshold: {args.reward_threshold}")
            print("=" * 60)
        if args.device is None:
            if torch.cuda.is_available():
                try:
                    torch.zeros(1).cuda()
                    device = "cuda"
                    print("Using GPU")
                except Exception as e:
                    print(f"Warning: CUDA unavailable: {e}, using CPU")
                    device = "cpu"
            else:
                device = "cpu"
                print("Using CPU")
        else:
            device = args.device
        print(f"Device: {device}")
        print(f"\nLoading model: {mask_path(args.model_path)}")
        try:
            from transformers import AutoProcessor
        except Exception as e:
            print(f"Warning: AutoProcessor import failed: {e}, trying Qwen2VLProcessor...")
            try:
                from transformers import Qwen2VLProcessor as AutoProcessor
            except ImportError:
                from transformers import Qwen2_5_VLProcessor as AutoProcessor
        
        try:
            from transformers import Qwen2_5_VLForConditionalGeneration
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                args.model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto" if device == "cuda" else "cpu",
                trust_remote_code=True
            )
            model.eval()
            processor = AutoProcessor.from_pretrained(args.model_path)
            print("Loaded Qwen2_5_VL")
        except ImportError:
            try:
                from transformers import Qwen2VLForConditionalGeneration
                model = Qwen2VLForConditionalGeneration.from_pretrained(
                    args.model_path,
                    torch_dtype=torch.bfloat16,
                    device_map="auto" if device == "cuda" else "cpu",
                    trust_remote_code=True
                )
                model.eval()
                try:
                    from transformers import Qwen2VLProcessor
                    processor = Qwen2VLProcessor.from_pretrained(args.model_path)
                except Exception:
                    processor = AutoProcessor.from_pretrained(args.model_path)
                print("Loaded Qwen2VL")
            except Exception as e:
                print(f"Qwen2VL load failed: {e}")
                raise
        except Exception as e:
            print(f"Model load failed: {e}")
            import traceback
            traceback.print_exc()
            raise
    
        print("\nInitializing reward model...")
        reward_model = VLRewardModel(
            model=model,
            tokenizer=processor.tokenizer,
            device=device,
            data_name=args.dataset,
            rule_format_string=None,
            ground_truth_weight=args.ground_truth_weight,
            dataset=args.dataset
        )
        if args.dataset == "mathvision":
            print("\nLoading MathVision...")
            all_data = load_mathvision_dataset(cache_dir=args.cache_dir, split="testmini")
            dataset_prefix = "mathvision"
        else:
            print("\nLoading MMMU...")
            all_data = load_mmmu_dataset(cache_dir=args.cache_dir, subsets=args.subsets)
            dataset_prefix = "mmmu"
        if not all_data:
            print("Error: no data loaded")
            return
        print(f"Loaded {len(all_data)} samples")
        start_idx = max(0, args.start_idx)
        end_idx = args.end_idx if args.end_idx is not None else len(all_data)
        end_idx = min(end_idx, len(all_data))
        print(f"Range: [{start_idx}, {end_idx})")
        os.makedirs(args.output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(args.output_dir, f"{dataset_prefix}_results_{timestamp}.jsonl")
        
        stats = {
            'total': 0,
            'success': 0,
            'failed': 0,
            'original_correct': 0,
            'optimized_correct': 0,
            'original_rewards': [],
            'optimized_rewards': [],
            'reward_improvements': []
        }
        pbar = tqdm(range(start_idx, end_idx), desc="Processing", unit="sample")
        for idx in pbar:
            sample = all_data[idx]
            stats['total'] += 1
            
            subset_name = sample.get('subset', 'unknown')
            pbar.set_postfix({
                'subset': subset_name[:15],
                'success': stats['success'],
                'orig_correct': stats.get('original_correct', 0),
                'opt_correct': stats.get('optimized_correct', 0)
            })
            
            result = process_single_sample(
                sample=sample,
                model=model,
                processor=processor,
                reward_model=reward_model,
                device=device,
                args=args
            )
            
            if result:
                stats['success'] += 1
                stats['original_rewards'].append(result['original_reward'])
                stats['optimized_rewards'].append(result['optimized_reward'])
                stats['reward_improvements'].append(
                    result['optimized_reward'] - result['original_reward']
                )
                
                if result.get('original_is_correct', False):
                    stats['original_correct'] = stats.get('original_correct', 0) + 1
                if result.get('optimized_is_correct', False):
                    stats['optimized_correct'] = stats.get('optimized_correct', 0) + 1
                pbar.set_postfix({
                    'subset': subset_name[:15],
                    'success': stats['success'],
                    'orig_correct': stats.get('original_correct', 0),
                    'opt_correct': stats.get('optimized_correct', 0)
                })
                with open(output_file, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(result, ensure_ascii=False) + '\n')
            else:
                stats['failed'] += 1
        print(f"\n{'='*60}")
        print("Done")
        print(f"{'='*60}")
        print(f"Total: {stats['total']}, Success: {stats['success']}, Failed: {stats['failed']}")
        print(f"Original correct: {stats.get('original_correct', 0)}, Optimized correct: {stats.get('optimized_correct', 0)}")
        if stats['total'] > 0:
            print(f"Original accuracy: {stats.get('original_correct', 0) / stats['total']:.4f}")
            print(f"Optimized accuracy: {stats.get('optimized_correct', 0) / stats['total']:.4f}")
        if stats['success'] > 0:
            print(f"Original (success): {stats.get('original_correct', 0) / stats['success']:.4f}")
            print(f"Optimized (success): {stats.get('optimized_correct', 0) / stats['success']:.4f}")
            print(f"Avg reward (orig/opt): {sum(stats['original_rewards'])/len(stats['original_rewards']):.4f} / {sum(stats['optimized_rewards'])/len(stats['optimized_rewards']):.4f}")
            print(f"Improved count: {sum(1 for imp in stats['reward_improvements'] if imp > 0)}")
        print(f"Results: {mask_path(output_file)}")
        print(f"Log: {mask_path(log_file)}")
        if len(training_data_list) > 0:
            dataset_name = args.dataset
            training_data_path = os.path.join(args.output_dir, f'vl_training_data_{dataset_name}_new.json')
            save_training_data(training_data_list, training_data_path)
            from collect_pre_segment3_data import validate_training_data
            validate_training_data(training_data_path)
        else:
            print("\nWarning: no training data collected")
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        if 'log_f' in locals():
            log_f.close()


_PATH_ARGS = ("model_path", "output_dir", "cache_dir", "log_file")


if __name__ == "__main__":
    args = parse_args()
    print("Config:")
    for arg in vars(args):
        val = getattr(args, arg)
        if arg in _PATH_ARGS and val is not None and str(val).strip():
            print(f"  --{arg}: {mask_path(val)}")
        else:
            print(f"  --{arg}: {val}")
    main(args)


