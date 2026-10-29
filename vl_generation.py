import torch
from typing import List, Tuple, Optional
from PIL import Image
import re
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'qwen-vl-utils', 'src'))
from qwen_vl_utils import process_vision_info
from vision_config import get_vision_resolution

stop_words = ["</s>", "<|im_end|>", "<|endoftext|>"]


def original_vl_generation(
    image: Image.Image,
    question: str,
    model,
    processor,
    device: str,
    max_new_tokens: int = 1024,
    dataset: str = "mmmu"
) -> Tuple[str, List[torch.Tensor], torch.Tensor, List[int]]:
    eos_token = processor.tokenizer.eos_token
    if eos_token:
        stop_words.append(eos_token)
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
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages,
        image_patch_size=processor.image_processor.patch_size if hasattr(processor, 'image_processor') else 14,
        return_video_kwargs=True,
        return_video_metadata=True
    )
    inputs = processor(
        text=[text],
        images=image_inputs if image_inputs else None,
        videos=None,
        padding=True,
        return_tensors="pt"
    ).to(device)
    input_length = inputs["input_ids"].shape[1]
    answer = []
    generated_token_ids = []
    hidden_states_list = []
    new_input_ids = inputs["input_ids"].clone()
    current_attention_mask = inputs.get("attention_mask")
    if current_attention_mask is not None:
        current_attention_mask = current_attention_mask.clone()
    cnt = 0
    vision_inputs = {}
    if "pixel_values" in inputs:
        vision_inputs["pixel_values"] = inputs["pixel_values"]
    if "image_grid_thw" in inputs:
        vision_inputs["image_grid_thw"] = inputs["image_grid_thw"]
    if "video_grid_thw" in inputs:
        vision_inputs["video_grid_thw"] = inputs["video_grid_thw"]
    
    while True:
        with torch.no_grad():
            model_inputs = {"input_ids": new_input_ids}
            if current_attention_mask is not None:
                model_inputs["attention_mask"] = current_attention_mask
            if cnt == 0:
                model_inputs.update(vision_inputs)
            model_inputs["output_hidden_states"] = True
            if hasattr(model, 'model'):
                outputs = model.model(**model_inputs)
            else:
                outputs = model(**model_inputs)
        if hasattr(outputs, 'hidden_states') and outputs.hidden_states:
            hidden_states = outputs.hidden_states[-1][:, -1]
        else:
            hidden_states = outputs.last_hidden_state[:, -1]
        hidden_states_list.append(hidden_states.clone())
        
        hidden_states = hidden_states.detach()
        hidden_states.requires_grad = True
        
        if hidden_states.grad is not None:
            hidden_states.grad.zero_()
        with torch.no_grad():
            if hasattr(model, 'lm_head'):
                eval_logits = model.lm_head(hidden_states)
            else:
                eval_logits = model.get_output_embeddings()(hidden_states)
            next_token_id = torch.argmax(eval_logits, dim=-1)
            new_token = processor.tokenizer.decode(next_token_id.item())
            token_id = next_token_id.item()
            answer.append(token_id)
            generated_token_ids.append(token_id)
            
            if new_token in stop_words:
                break
            new_input_ids = torch.cat([new_input_ids, next_token_id.unsqueeze(0)], dim=-1)
            if current_attention_mask is not None:
                new_attention = torch.ones((current_attention_mask.shape[0], 1), 
                                          dtype=current_attention_mask.dtype, 
                                          device=current_attention_mask.device)
                current_attention_mask = torch.cat([current_attention_mask, new_attention], dim=-1)
        
        cnt += 1
        if cnt > max_new_tokens:
            break
    
    answer_text = processor.tokenizer.decode(answer, skip_special_tokens=True)
    return answer_text, hidden_states_list, new_input_ids, generated_token_ids


def find_sentence_boundaries(text: str) -> List[int]:
    boundaries = []
    for match in re.finditer(r'[.\n]', text):
        boundaries.append(match.start())
    
    return boundaries


def extract_latent_segments(
    hidden_states_list: List[torch.Tensor],
    generated_text: str,
    token_ids: List[int],
    tokenizer,
    segment_length: int = 10,
    use_percentage: bool = False,
    percentage: float = 0.05
) -> List[Tuple[int, int, List[torch.Tensor]]]:
    segments = []
    
    if len(hidden_states_list) == 0:
        return segments
    
    total_length = len(hidden_states_list)
    if use_percentage:
        actual_segment_length = max(1, int(total_length * percentage))
    else:
        actual_segment_length = segment_length
    first_segment_end = min(actual_segment_length, total_length)
    if first_segment_end > 0:
        segments.append((0, first_segment_end, hidden_states_list[:first_segment_end]))
    sentence_boundaries = find_sentence_boundaries(generated_text)
    if len(sentence_boundaries) == 0:
        return segments
    decoded_text_so_far = ""
    token_positions = []
    for i, token_id in enumerate(token_ids):
        token_text = tokenizer.decode([token_id], skip_special_tokens=True)
        decoded_text_so_far += token_text
        token_positions.append(len(decoded_text_so_far))
    first_boundary_text_pos = sentence_boundaries[0]
    first_boundary_token_pos = 0
    for i, text_pos in enumerate(token_positions):
        if text_pos > first_boundary_text_pos:
            first_boundary_token_pos = i
            break
    first_boundary_token_pos = min(first_boundary_token_pos, total_length - 1)
    second_segment_start = first_boundary_token_pos + 1
    second_segment_end = min(second_segment_start + actual_segment_length, total_length)
    if second_segment_start < total_length and second_segment_end > second_segment_start:
        segments.append((second_segment_start, second_segment_end,
                        hidden_states_list[second_segment_start:second_segment_end]))
    if len(sentence_boundaries) >= 2:
        second_boundary_text_pos = sentence_boundaries[1]
        second_boundary_token_pos = 0
        for i, text_pos in enumerate(token_positions):
            if text_pos > second_boundary_text_pos:
                second_boundary_token_pos = i
                break
        
        second_boundary_token_pos = min(second_boundary_token_pos, total_length - 1)
        third_segment_start = second_boundary_token_pos + 1
        third_segment_end = min(third_segment_start + actual_segment_length, total_length)
        if third_segment_start < total_length and third_segment_end > third_segment_start:
            segments.append((third_segment_start, third_segment_end,
                            hidden_states_list[third_segment_start:third_segment_end]))
    
    return segments



