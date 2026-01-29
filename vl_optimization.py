import torch
from typing import List, Tuple, Optional, Any, Dict
from PIL import Image
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'qwen-vl-utils', 'src'))
from qwen_vl_utils import process_vision_info
from vision_config import get_vision_resolution


def optimize_latent_segment(
    segment_hidden_states: List[torch.Tensor],
    start_idx: int,
    end_idx: int,
    reward_model: Any,  # MMMURewardModel
    model,
    processor,
    device: str,
    question: str,
    original_answer: str,
    original_input_ids: torch.Tensor,
    base_input_ids: torch.Tensor,
    image=None,  # PIL.Image for multimodal verification
    max_num_steps: int = 10,
    lr: float = 0.03,
    grad_clip: Optional[float] = None,
    reward_threshold: float = -0.2,
    max_new_tokens: int = 2048,
    ground_truth=None,  # Ground truth answer for reward calculation
    has_options=True,  # Whether the question has options
    use_cold_start: bool = True,
    dataset: str = "mmmu"
) -> Tuple[str, List[float], int, List[torch.Tensor]]:
    segment_length = len(segment_hidden_states)
    if segment_length == 0:
        return original_answer, [0.0], 0, []
    
    processed_image_inputs = None
    if image is not None:
        MIN_PIXELS, MAX_PIXELS = get_vision_resolution(dataset)
        
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image,
                        "min_pixels": MIN_PIXELS,
                        "max_pixels": MAX_PIXELS
                    },
                    {"type": "text", "text": question}
                ]
            }
        ]
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages,
            image_patch_size=processor.image_processor.patch_size if hasattr(processor, 'image_processor') else 14,
            return_video_kwargs=True,
            return_video_metadata=True
        )
        processed_image_inputs = image_inputs
    optimized_hidden_states = torch.nn.Parameter(torch.stack([
        state.clone().detach().requires_grad_(True)
        for state in segment_hidden_states
    ]))
    original_hidden_states = torch.stack([
        state.clone().detach() for state in segment_hidden_states
    ])
    optimizer = torch.optim.Adam([optimized_hidden_states], lr=lr)
    reward_history = []
    if use_cold_start:
        print("\n" + "="*80)
        print("COLD-START (Round 0): ground-truth reward")
        print("="*80)
        print(f"Original: {original_answer[:200]}..." if len(original_answer) > 200 else f"Original: {original_answer}")
        if ground_truth:
            print(f"Ground Truth: {ground_truth}")
        print("-"*80)
        initial_reward = reward_model.get_reward(question, original_answer, image=image, processor=processor, ground_truth=ground_truth, has_options=has_options, optimization_round=0)
        reward_history.append(initial_reward)
        current_reward = initial_reward
        print("-"*80)
        print(f"COLD-START reward: {initial_reward:.4f}")
        print("="*80 + "\n")
    else:
        print("\n" + "="*80)
        print("TTS (Round 0): no ground-truth reward")
        print("="*80)
        print(f"Original: {original_answer[:200]}..." if len(original_answer) > 200 else f"Original: {original_answer}")
        print("-"*80)
        initial_reward = reward_model.get_reward(question, original_answer, image=image, processor=processor, ground_truth=None, has_options=has_options, optimization_round=1)
        reward_history.append(initial_reward)
        current_reward = initial_reward
        print("-"*80)
        print(f"TTS Round 0 reward: {initial_reward:.4f}")
        print("="*80 + "\n")
    best_reward = initial_reward
    best_answer = original_answer
    best_hidden_states = optimized_hidden_states.clone()
    patience = 2
    no_improvement_count = 0
    same_answer_count = 0
    last_answer = original_answer
    last_hidden_states_hash = None
    new_answer = None
    if use_cold_start:
        print("\n" + "="*80)
        print("TTS: test-time self-supervised optimization")
        print("="*80 + "\n")
    for step in range(max_num_steps):
        if step > 0 and step % 3 == 0:
            print(f"  Step {step}/{max_num_steps}, reward {current_reward:.4f}, best {best_reward:.4f}")
        if step >= 3 and no_improvement_count >= patience:
            print(f"  No improvement for {patience} steps, early stop at {step+1}")
            optimized_hidden_states.data = best_hidden_states
            optimized_length = len(processor.tokenizer.encode(best_answer))
            optimized_hidden_states_list = [optimized_hidden_states[i].clone().detach() for i in range(segment_length)]
            return best_answer, reward_history, optimized_length, optimized_hidden_states_list
        if current_reward > reward_threshold:
            final_answer = new_answer if new_answer is not None else original_answer
            optimized_length = len(processor.tokenizer.encode(final_answer))
            optimized_hidden_states_list = [optimized_hidden_states[i].clone().detach() for i in range(segment_length)]
            return final_answer, reward_history, optimized_length, optimized_hidden_states_list
        current_hidden_states_hash = optimized_hidden_states.mean().item()
        if last_hidden_states_hash is not None:
            if abs(current_hidden_states_hash - last_hidden_states_hash) < 0.001:
                pass
        last_hidden_states_hash = current_hidden_states_hash
        optimizer.zero_grad()
        if hasattr(model, 'lm_head'):
            logits = model.lm_head(optimized_hidden_states)
        else:
            logits = model.get_output_embeddings()(optimized_hidden_states)
        
        probs = torch.softmax(logits, dim=-1) + 1e-8
        next_token_ids = torch.argmax(probs, dim=-1)
        next_token_ids = next_token_ids.squeeze(-1)
        log_pi_xz = torch.log(
            probs[torch.arange(segment_length), 0, next_token_ids] + 1e-10
        )
        loss = -current_reward * log_pi_xz.sum()
        loss.backward(retain_graph=True)
        if grad_clip:
            torch.nn.utils.clip_grad_norm_([optimized_hidden_states], grad_clip)
        else:
            torch.nn.utils.clip_grad_norm_([optimized_hidden_states], 1.0)
        optimizer.step()
        input_ids = base_input_ids.clone()
        with torch.no_grad():
            if hasattr(model, 'lm_head'):
                next_tokens = torch.argmax(model.lm_head(optimized_hidden_states), dim=-1)
            else:
                next_tokens = torch.argmax(model.get_output_embeddings()(optimized_hidden_states), dim=-1)
            next_tokens = next_tokens.squeeze(-1)
            for token_id in next_tokens:
                token_id_tensor = token_id.unsqueeze(0).unsqueeze(0)
                input_ids = torch.cat([input_ids, token_id_tensor], dim=-1)
        with torch.no_grad():
            cnt = 0
            max_generation_steps = max_new_tokens
            stop_tokens = ["</s>", "<|im_end|>", "<|endoftext|>"]
            eos_token = processor.tokenizer.eos_token
            if eos_token:
                stop_tokens.append(eos_token)
            min_generation_length = 15
            recent_tokens = []
            max_recent_length = 10
            repeat_count = 0
            while cnt < max_generation_steps:
                if cnt == 0 and processed_image_inputs is not None:
                    vision_outputs = processor.image_processor(
                        images=processed_image_inputs,
                        return_tensors="pt"
                    )
                    model_forward_inputs = {
                        "input_ids": input_ids,
                        "output_hidden_states": True
                    }
                    for key in vision_outputs:
                        if isinstance(vision_outputs[key], torch.Tensor):
                            model_forward_inputs[key] = vision_outputs[key].to(device)
                        else:
                            model_forward_inputs[key] = vision_outputs[key]
                else:
                    model_forward_inputs = {
                        "input_ids": input_ids,
                        "output_hidden_states": True
                    }
                
                if hasattr(model, 'model'):
                    outputs = model.model(**model_forward_inputs)
                else:
                    outputs = model(**model_forward_inputs)
                if hasattr(outputs, 'hidden_states'):
                    hidden_states = outputs.hidden_states[-1][:, -1]
                elif hasattr(outputs, 'last_hidden_state'):
                    hidden_states = outputs.last_hidden_state[:, -1]
                else:
                    hidden_states = outputs[0][:, -1] if isinstance(outputs, tuple) else outputs[:, -1]
                if hasattr(model, 'lm_head'):
                    logits = model.lm_head(hidden_states)
                else:
                    logits = model.get_output_embeddings()(hidden_states)
                if len(recent_tokens) >= max_recent_length:
                    recent_tokens.pop(0)
                
                next_token_id = torch.argmax(logits, dim=-1)
                next_token_id_val = next_token_id.item() if next_token_id.numel() == 1 else next_token_id[0].item()
                if len(recent_tokens) > 0 and next_token_id_val == recent_tokens[-1]:
                    repeat_count += 1
                    if repeat_count >= 3:
                        temperature = 0.8
                        probs = torch.softmax(logits / temperature, dim=-1)
                        top_k = 5
                        top_k_probs, top_k_indices = torch.topk(probs, top_k, dim=-1)
                        sampled_idx = torch.multinomial(top_k_probs.squeeze(0), 1)
                        next_token_id = top_k_indices.squeeze(0)[sampled_idx]
                        if next_token_id.dim() > 0:
                            next_token_id = next_token_id[0]
                        next_token_id_val = next_token_id.item()
                        repeat_count = 0
                else:
                    repeat_count = 0
                
                recent_tokens.append(next_token_id_val)
                new_token = processor.tokenizer.decode(next_token_id_val)
                if isinstance(next_token_id, torch.Tensor):
                    if next_token_id.dim() > 0:
                        next_token_id = next_token_id[0]
                    next_token_id_tensor = next_token_id.unsqueeze(0).unsqueeze(0)
                else:
                    next_token_id_tensor = torch.tensor([[next_token_id_val]], device=input_ids.device)
                input_ids = torch.cat([input_ids, next_token_id_tensor], dim=-1)
                cnt += 1
                if (new_token == eos_token or new_token in stop_tokens) and cnt >= min_generation_length:
                    break
                if cnt >= max_generation_steps:
                    break
        generated_ids = input_ids[0, base_input_ids.shape[1]:]
        new_answer = processor.tokenizer.decode(generated_ids, skip_special_tokens=True)
        new_answer_stripped = new_answer.strip() if new_answer else ""
        last_answer_stripped = last_answer.strip() if last_answer else ""
        if new_answer_stripped == last_answer_stripped and len(new_answer_stripped) > 0:
            print(f"  Answer unchanged, stop at step {step+1}")
            optimized_hidden_states.data = best_hidden_states
            optimized_length = len(processor.tokenizer.encode(best_answer))
            optimized_hidden_states_list = [optimized_hidden_states[i].clone().detach() for i in range(segment_length)]
            return best_answer, reward_history, optimized_length, optimized_hidden_states_list
        same_answer_count = 0
        last_answer = new_answer
        optimization_round = step + 1 if use_cold_start else step + 2
        print(f"\nTTS Round {step+1}/{max_num_steps}: computing reward")
        print(f"New answer: {new_answer[:200]}..." if len(new_answer) > 200 else f"New answer: {new_answer}")
        current_reward = reward_model.get_reward(question, new_answer, image=image, processor=processor, ground_truth=None, has_options=has_options, optimization_round=optimization_round)
        reward_history.append(current_reward)
        print(f"TTS Round {step+1} reward: {current_reward:.4f} (best: {best_reward:.4f})")
        if current_reward > best_reward:
            best_reward = current_reward
            best_answer = new_answer
            best_hidden_states = optimized_hidden_states.clone()
            no_improvement_count = 0
        else:
            no_improvement_count += 1
            if no_improvement_count >= patience:
                optimized_hidden_states.data = best_hidden_states
                optimized_length = len(processor.tokenizer.encode(best_answer))
                optimized_hidden_states_list = [optimized_hidden_states[i].clone().detach() for i in range(segment_length)]
                return best_answer, reward_history, optimized_length, optimized_hidden_states_list
        torch.cuda.empty_cache()
    final_answer = best_answer if best_answer is not None else original_answer
    optimized_length = len(processor.tokenizer.encode(final_answer))
    optimized_hidden_states.data = best_hidden_states
    optimized_hidden_states_list = [optimized_hidden_states[i].clone().detach() for i in range(segment_length)]
    
    return final_answer, reward_history, optimized_length, optimized_hidden_states_list


def generate_answer_with_hidden_states(
    base_input_ids: torch.Tensor,
    optimized_segment_hidden_states: List[torch.Tensor],
    model,
    processor,
    device: str,
    max_new_tokens: int = 2048,
    processed_image_inputs: Optional[list] = None
) -> Tuple[str, torch.Tensor, List[torch.Tensor], List[int]]:
    input_ids = base_input_ids.clone()
    all_hidden_states = []
    generated_token_ids = []
    stop_tokens = ["</s>", "<|im_end|>", "<|endoftext|>"]
    eos_token = processor.tokenizer.eos_token
    if eos_token:
        stop_tokens.append(eos_token)
    with torch.no_grad():
        if hasattr(model, 'lm_head'):
            segment_tokens = torch.argmax(model.lm_head(torch.stack(optimized_segment_hidden_states)), dim=-1)
        else:
            segment_tokens = torch.argmax(model.get_output_embeddings()(torch.stack(optimized_segment_hidden_states)), dim=-1)
        segment_tokens = segment_tokens.squeeze(-1) if segment_tokens.dim() > 1 else segment_tokens
        for token_id in segment_tokens:
            token_id_tensor = token_id.unsqueeze(0).unsqueeze(0) if token_id.dim() == 0 else token_id.unsqueeze(0)
            input_ids = torch.cat([input_ids, token_id_tensor], dim=-1)
            generated_token_ids.append(token_id.item() if hasattr(token_id, 'item') else int(token_id))
        all_hidden_states.extend(optimized_segment_hidden_states)
    cnt = 0
    min_generation_length = 15
    recent_tokens = []
    max_recent_length = 10
    repeat_count = 0
    
    while cnt < max_new_tokens:
        with torch.no_grad():
            if cnt == 0 and processed_image_inputs is not None:
                vision_outputs = processor.image_processor(
                    images=processed_image_inputs,
                    return_tensors="pt"
                )
                model_forward_inputs = {
                    "input_ids": input_ids,
                    "output_hidden_states": True
                }
                for key in vision_outputs:
                    if isinstance(vision_outputs[key], torch.Tensor):
                        model_forward_inputs[key] = vision_outputs[key].to(device)
                    else:
                        model_forward_inputs[key] = vision_outputs[key]
            else:
                model_forward_inputs = {
                    "input_ids": input_ids,
                    "output_hidden_states": True
                }
            
            if hasattr(model, 'model'):
                outputs = model.model(**model_forward_inputs)
            else:
                outputs = model(**model_forward_inputs)
            if hasattr(outputs, 'hidden_states'):
                hidden_states = outputs.hidden_states[-1][:, -1]
            elif hasattr(outputs, 'last_hidden_state'):
                hidden_states = outputs.last_hidden_state[:, -1]
            else:
                hidden_states = outputs[0][:, -1] if isinstance(outputs, tuple) else outputs[:, -1]
            all_hidden_states.append(hidden_states.clone())
            if hasattr(model, 'lm_head'):
                logits = model.lm_head(hidden_states)
            else:
                logits = model.get_output_embeddings()(hidden_states)
            if len(recent_tokens) >= max_recent_length:
                recent_tokens.pop(0)
            next_token_id = torch.argmax(logits, dim=-1)
            next_token_id_val = next_token_id.item() if next_token_id.numel() == 1 else next_token_id[0].item()
            if len(recent_tokens) > 0 and next_token_id_val == recent_tokens[-1]:
                repeat_count += 1
                if repeat_count >= 3:
                    temperature = 0.8
                    probs = torch.softmax(logits / temperature, dim=-1)
                    top_k = 5
                    top_k_probs, top_k_indices = torch.topk(probs, top_k, dim=-1)
                    sampled_idx = torch.multinomial(top_k_probs.squeeze(0), 1)
                    next_token_id = top_k_indices.squeeze(0)[sampled_idx]
                    if next_token_id.dim() > 0:
                        next_token_id = next_token_id[0]
                    next_token_id_val = next_token_id.item()
                    repeat_count = 0
            else:
                repeat_count = 0
            
            recent_tokens.append(next_token_id_val)
            new_token = processor.tokenizer.decode(next_token_id_val)
            if isinstance(next_token_id, torch.Tensor):
                if next_token_id.dim() > 0:
                    next_token_id = next_token_id[0]
                next_token_id_tensor = next_token_id.unsqueeze(0).unsqueeze(0)
            else:
                next_token_id_tensor = torch.tensor([[next_token_id_val]], device=input_ids.device)
            input_ids = torch.cat([input_ids, next_token_id_tensor], dim=-1)
            generated_token_ids.append(next_token_id_val)
            cnt += 1
            if new_token in stop_tokens and cnt >= min_generation_length:
                break
    generated_ids = input_ids[0, base_input_ids.shape[1]:]
    answer_text = processor.tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    return answer_text, input_ids, all_hidden_states, generated_token_ids


def optimize_all_segments(
    segments: List[Tuple[int, int, List[torch.Tensor]]],
    reward_model: Any,  # MMMURewardModel
    model,
    processor,
    device: str,
    question: str,
    original_answer: str,
    original_input_ids: torch.Tensor,
    base_input_ids: torch.Tensor,
    image=None,  # PIL.Image for multimodal verification
    max_num_steps: int = 10,
    lr: float = 0.03,
    grad_clip: Optional[float] = None,
    reward_threshold: float = -0.2,
    max_new_tokens: int = 2048,
    ground_truth=None,  # Ground truth answer for reward calculation
    has_options=True,  # Whether the question has options
    dataset: str = "mmmu"  # Dataset type for resolution config
) -> Tuple[str, List[List[float]], int, List[Tuple[int, int, List[torch.Tensor], torch.Tensor]], List[torch.Tensor]]:
    if not segments:
        return original_answer, [], 0, [], None
    
    all_reward_histories = []
    segment_results = []
    total_optimized_length = 0
    from vl_generation import extract_latent_segments
    processed_image_inputs = None
    if image is not None:
        MIN_PIXELS, MAX_PIXELS = get_vision_resolution(dataset)
        
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image,
                        "min_pixels": MIN_PIXELS,
                        "max_pixels": MAX_PIXELS
                    },
                    {"type": "text", "text": question}
                ]
            }
        ]
        
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages,
            image_patch_size=processor.image_processor.patch_size if hasattr(processor, 'image_processor') else 14,
            return_video_kwargs=True,
            return_video_metadata=True
        )
        processed_image_inputs = image_inputs
    first_seg_start, first_seg_end, first_seg_hidden_states = segments[0]
    print(f"\nSegment 1/3 ({first_seg_start}-{first_seg_end})")
    first_optimized_answer, first_reward_history, first_optimized_length, first_optimized_hidden_states = optimize_latent_segment(
        segment_hidden_states=first_seg_hidden_states,
        start_idx=first_seg_start,
        end_idx=first_seg_end,
        reward_model=reward_model,
        model=model,
        processor=processor,
        device=device,
        question=question,
        original_answer=original_answer,
        original_input_ids=original_input_ids,
        base_input_ids=base_input_ids,
        image=image,
        max_num_steps=max_num_steps,
        lr=lr,
        grad_clip=grad_clip,
        reward_threshold=reward_threshold,
        max_new_tokens=max_new_tokens,
        ground_truth=ground_truth,
        has_options=has_options,
        use_cold_start=True,
        dataset=dataset
    )
    all_reward_histories.append(first_reward_history)
    total_optimized_length += first_optimized_length
    segment_results.append((first_seg_start, first_seg_end, first_optimized_hidden_states, base_input_ids.clone()))
    first_regenerated_answer, first_optimized_input_ids, first_all_hidden_states, first_generated_token_ids = generate_answer_with_hidden_states(
        base_input_ids=base_input_ids,
        optimized_segment_hidden_states=first_optimized_hidden_states,
        model=model,
        processor=processor,
        device=device,
        max_new_tokens=max_new_tokens,
        processed_image_inputs=processed_image_inputs
    )
    second_segments = extract_latent_segments(
        hidden_states_list=first_all_hidden_states,
        generated_text=first_regenerated_answer,
        token_ids=first_generated_token_ids,
        tokenizer=processor.tokenizer,
        segment_length=20,
        use_percentage=True,
        percentage=0.05
    )
    if len(second_segments) < 2:
        print("No second segment after first, skipping 2 & 3")
        return first_optimized_answer, all_reward_histories, total_optimized_length, segment_results, first_all_hidden_states
    second_seg_start, second_seg_end, second_seg_hidden_states = second_segments[1]
    print(f"\nSegment 2/3 ({second_seg_start}-{second_seg_end})")
    second_base_input_ids = first_optimized_input_ids[:, :base_input_ids.shape[1] + second_seg_start]
    second_optimized_answer, second_reward_history, second_optimized_length, second_optimized_hidden_states = optimize_latent_segment(
        segment_hidden_states=second_seg_hidden_states,
        start_idx=second_seg_start,
        end_idx=second_seg_end,
        reward_model=reward_model,
        model=model,
        processor=processor,
        device=device,
        question=question,
        original_answer=first_optimized_answer,
        original_input_ids=first_optimized_input_ids,
        base_input_ids=second_base_input_ids,
        image=image,
        max_num_steps=max_num_steps,
        lr=lr,
        grad_clip=grad_clip,
        reward_threshold=reward_threshold,
        max_new_tokens=max_new_tokens,
        ground_truth=None,
        has_options=has_options,
        use_cold_start=False,
        dataset=dataset
    )
    
    all_reward_histories.append(second_reward_history)
    total_optimized_length += second_optimized_length
    segment_results.append((
        second_seg_start,
        second_seg_end,
        second_optimized_hidden_states,
        second_base_input_ids.clone()
    ))
    second_regenerated_answer, second_optimized_input_ids, second_all_hidden_states, second_generated_token_ids = generate_answer_with_hidden_states(
        base_input_ids=second_base_input_ids,
        optimized_segment_hidden_states=second_optimized_hidden_states,
        model=model,
        processor=processor,
        device=device,
        max_new_tokens=max_new_tokens,
        processed_image_inputs=processed_image_inputs
    )
    third_segments = extract_latent_segments(
        hidden_states_list=second_all_hidden_states,
        generated_text=second_regenerated_answer,
        token_ids=second_generated_token_ids,
        tokenizer=processor.tokenizer,
        segment_length=20,
        use_percentage=True,
        percentage=0.05
    )
    if len(third_segments) < 3:
        print("No third segment after second, skipping")
        return second_optimized_answer, all_reward_histories, total_optimized_length, segment_results, second_all_hidden_states
    third_seg_start, third_seg_end, third_seg_hidden_states = third_segments[2]
    print(f"\nSegment 3/3 ({third_seg_start}-{third_seg_end})")
    third_base_input_ids = second_optimized_input_ids[:, :second_base_input_ids.shape[1] + third_seg_start]
    third_optimized_answer, third_reward_history, third_optimized_length, third_optimized_hidden_states = optimize_latent_segment(
        segment_hidden_states=third_seg_hidden_states,
        start_idx=third_seg_start,
        end_idx=third_seg_end,
        reward_model=reward_model,
        model=model,
        processor=processor,
        device=device,
        question=question,
        original_answer=second_optimized_answer,
        original_input_ids=second_optimized_input_ids,
        base_input_ids=third_base_input_ids,
        image=image,
        max_num_steps=max_num_steps,
        lr=lr,
        grad_clip=grad_clip,
        reward_threshold=reward_threshold,
        max_new_tokens=max_new_tokens,
        ground_truth=None,
        has_options=has_options,
        use_cold_start=False,
        dataset=dataset
    )
    
    all_reward_histories.append(third_reward_history)
    total_optimized_length += third_optimized_length
    segment_results.append((
        third_seg_start,
        third_seg_end,
        third_optimized_hidden_states,
        third_base_input_ids.clone()
    ))
    return third_optimized_answer, all_reward_histories, total_optimized_length, segment_results, second_all_hidden_states

