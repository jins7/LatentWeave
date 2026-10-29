import json
import os
import base64
from io import BytesIO
from PIL import Image
from typing import List, Dict, Any, Tuple
import torch

from utils import mask_path


def collect_pre_segment3_data(
    sample: Dict[str, Any],
    original_hidden_states_list: List[torch.Tensor],
    original_segments: List[Tuple[int, int, List[torch.Tensor]]],
    segment_results: List[Tuple[int, int, List[torch.Tensor], torch.Tensor]],
    image: Image.Image,
    question_with_options: str,
    optimized_is_correct: bool,
    optimized_hidden_states_list: List[torch.Tensor] = None
) -> Dict[str, Any]:
    if not optimized_is_correct:
        return None
    if len(original_segments) < 3 or len(segment_results) < 3:
        return None
    third_seg_start = original_segments[2][0]
    orig_latent_pre_seg3 = original_hidden_states_list[:third_seg_start]
    if optimized_hidden_states_list is not None:
        opt_latent_pre_seg3 = optimized_hidden_states_list[:third_seg_start]
    else:
        opt_latent_pre_seg3 = []
        if len(opt_latent_pre_seg3) == 0:
            print("Warning: need optimized_hidden_states_list to rebuild from segment_results")
            return None
    if len(orig_latent_pre_seg3) != len(opt_latent_pre_seg3):
        print(f"Warning: pre-seg3 latent length mismatch ({len(orig_latent_pre_seg3)} vs {len(opt_latent_pre_seg3)})")
        return None
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
    context_input_ids = segment_results[0][3] if len(segment_results[0]) > 3 else None
    training_data = {
        'prompt': question_with_options,
        'image_base64': img_base64,
        'orig_latent': [lat.cpu().float().numpy().tolist() for lat in orig_latent_pre_seg3],
        'opt_latent': [lat.cpu().float().numpy().tolist() for lat in opt_latent_pre_seg3],
        'context_input_ids': context_input_ids.cpu().int().numpy().tolist() if context_input_ids is not None else None,
        'third_segment_start': int(third_seg_start),
        'total_length': len(original_hidden_states_list),
        'segment_info': {
            'segment_1': {
                'orig_start': int(original_segments[0][0]),
                'orig_end': int(original_segments[0][1]),
                'opt_start': int(segment_results[0][0]),
                'opt_end': int(segment_results[0][1])
            },
            'segment_2': {
                'orig_start': int(original_segments[1][0]),
                'orig_end': int(original_segments[1][1]),
                'opt_start': int(segment_results[1][0]),
                'opt_end': int(segment_results[1][1])
            },
            'segment_3_start': int(original_segments[2][0])
        },
        'subset': sample.get('subset', 'unknown'),
        'ground_truth': sample.get('answer', ''),
        'options': sample.get('options', [])
    }
    print(f"Collected pre-seg3: {len(orig_latent_pre_seg3)} tokens (seg3 starts at {third_seg_start})")
    return training_data


def collect_pre_segment3_from_regenerated(
    sample: Dict[str, Any],
    original_hidden_states_list: List[torch.Tensor],
    original_segments: List[Tuple[int, int, List[torch.Tensor]]],
    segment_results: List[Tuple[int, int, List[torch.Tensor], torch.Tensor]],
    image: Image.Image,
    question_with_options: str,
    optimized_is_correct: bool,
    second_regenerated_hidden_states: List[torch.Tensor] = None,
    original_answer: str = None,
    optimized_answer: str = None
) -> Dict[str, Any]:
    if not optimized_is_correct:
        return None
    if len(original_segments) < 3 or len(segment_results) < 3:
        return None
    third_seg_start_orig = original_segments[2][0]
    third_seg_end_orig = original_segments[2][1]
    orig_latent_with_seg3 = original_hidden_states_list[:third_seg_end_orig]
    if second_regenerated_hidden_states is None:
        print("Warning: second_regenerated_hidden_states required")
        return None
    if len(segment_results) >= 3:
        third_seg_end_opt = segment_results[2][1]
    else:
        third_seg_end_opt = third_seg_end_orig
    opt_latent_with_seg3 = second_regenerated_hidden_states[:third_seg_end_opt]
    if len(second_regenerated_hidden_states) < third_seg_end_opt:
        opt_latent_with_seg3 = second_regenerated_hidden_states
    if len(orig_latent_with_seg3) < third_seg_end_orig or len(opt_latent_with_seg3) < third_seg_end_opt:
        print(f"Warning: length insufficient orig {len(orig_latent_with_seg3)} (need {third_seg_end_orig}) opt {len(opt_latent_with_seg3)} (need {third_seg_end_opt})")
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
    context_input_ids = segment_results[0][3] if len(segment_results[0]) > 3 else None
    segment_start_indices = [
        int(original_segments[0][0]),
        int(original_segments[1][0]),
        int(original_segments[2][0])
    ]
    training_data = {
        'prompt': question_with_options,
        'image_base64': img_base64,
        'orig_latent': [lat.cpu().float().numpy().tolist() for lat in orig_latent_with_seg3],
        'opt_latent': [lat.cpu().float().numpy().tolist() for lat in opt_latent_with_seg3],
        'context_input_ids': context_input_ids.cpu().int().numpy().tolist() if context_input_ids is not None else None,
        'third_segment_start': int(third_seg_start_orig),
        'third_segment_end': int(third_seg_end_orig),
        'total_length': len(original_hidden_states_list),
        'segment_info': {
            'segment_1': {
                'orig_start': int(original_segments[0][0]),
                'orig_end': int(original_segments[0][1]),
                'opt_start': int(segment_results[0][0]),
                'opt_end': int(segment_results[0][1])
            },
            'segment_2': {
                'orig_start': int(original_segments[1][0]),
                'orig_end': int(original_segments[1][1]),
                'opt_start': int(segment_results[1][0]),
                'opt_end': int(segment_results[1][1])
            },
            'segment_3_start': int(original_segments[2][0])
        },
        'subset': sample.get('subset', 'unknown'),
        'ground_truth': sample.get('answer', ''),
        'options': sample.get('options', []),
        'original_answer': original_answer,
        'optimized_answer': optimized_answer,
        'segment_start_indices': segment_start_indices
    }
    print(f"Collected with seg3: orig {len(orig_latent_with_seg3)} opt {len(opt_latent_with_seg3)} tokens")
    return training_data


def save_training_data(training_data_list: List[Dict], output_path: str):
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(training_data_list, f, ensure_ascii=False, indent=2)
    print(f"\n{'='*60}")
    print(f"Saved to {mask_path(output_path)}, count {len(training_data_list)}")
    print(f"{'='*60}\n")


def validate_training_data(data_path: str):
    print(f"\n{'='*60}")
    print(f"Validating {mask_path(data_path)}")
    print(f"{'='*60}\n")
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    errors = []
    stats = {'total': len(data), 'dimensions': set(), 'lengths': []}
    for i, item in enumerate(data):
        required_fields = ['prompt', 'image_base64', 'orig_latent', 'opt_latent']
        missing = [f for f in required_fields if f not in item]
        if missing:
            errors.append(f"Sample {i}: missing {missing}")
            continue
        orig_latent = item['orig_latent']
        opt_latent = item['opt_latent']
        if len(orig_latent) != len(opt_latent):
            errors.append(f"Sample {i}: latent length mismatch ({len(orig_latent)} vs {len(opt_latent)})")
            continue
        if len(orig_latent) > 0:
            orig_dim = len(orig_latent[0])
            opt_dim = len(opt_latent[0])
            if orig_dim != opt_dim:
                errors.append(f"Sample {i}: latent dim mismatch ({orig_dim} vs {opt_dim})")
            else:
                stats['dimensions'].add(orig_dim)
                stats['lengths'].append(len(orig_latent))
    print(f"Total: {stats['total']}, dims: {sorted(stats['dimensions'])}")
    if stats['lengths']:
        import numpy as np
        print(f"Segment len: min={min(stats['lengths'])}, max={max(stats['lengths'])}, mean={np.mean(stats['lengths']):.1f}")
    if errors:
        print(f"\nErrors ({len(errors)}):")
        for e in errors[:10]:
            print(f"  - {e}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more")
    else:
        print("\nValidation passed")
    print(f"{'='*60}\n")
    return len(errors) == 0


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Validate training data JSON")
    parser.add_argument('--data_path', type=str, required=True, help='Path to training data JSON')
    args = parser.parse_args()
    validate_training_data(args.data_path)

