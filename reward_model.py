import os
import re
import sys
import torch
from termcolor import colored
from prompts import get_mmmu_verifier_prompt, VERA_ANSWER_SYMBOL
from utils import is_answer_correct, extract_answer_from_text
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'qwen-vl-utils', 'src'))
from qwen_vl_utils import process_vision_info
from vision_config import get_vision_resolution


def force_colored(text, color, attrs=None):
    color_codes = {
        'green': '\033[92m',
        'red': '\033[91m',
        'cyan': '\033[96m',
        'yellow': '\033[93m',
        'magenta': '\033[95m',
    }
    reset = '\033[0m'
    color_code = color_codes.get(color, '')
    attr_code = ''
    if attrs and 'bold' in attrs:
        attr_code = '\033[1m'
    return f"{color_code}{attr_code}{text}{reset}"


class VLRewardModel(object):
    def __init__(
        self,
        model,
        tokenizer,
        data_name: str = "mmmu",
        device: str = "cuda",
        rule_format_string: str = None,
        ground_truth_weight: float = 0,
        dataset: str = "mmmu",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.data_name = data_name 
        self.device = device
        self.rule_format_string = rule_format_string
        self.ground_truth_weight = ground_truth_weight
        self.dataset = dataset 

    def load_domain_specific_verifiers(self):
        veras = [
            "reasoning_check",
            "answer_correct",
            "answer_completeness",
            "understanding_check",
        ]
        return veras

    def get_verifications(self, question: str, solution: str, image=None, processor=None):
        '''
        Get verifications from different verifiers using dataset-specific prompts.

        Args:
            question: str, question (with options)
            solution: str, solution
            image: PIL.Image, optional image for multimodal verification
            processor: processor for handling multimodal inputs

        Returns:
            verifications: dict, verifier_name -> verifier_approval
        '''
        veras = self.load_domain_specific_verifiers()
        verifications = dict()
        for vera_type in veras:
            vera_prompt = get_mmmu_verifier_prompt(vera_type, question, solution)
            if image is not None and processor is not None:
                try:
                    MIN_PIXELS, MAX_PIXELS = get_vision_resolution(self.dataset)
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
                                {"type": "text", "text": vera_prompt}
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
                        padding=True,
                        return_tensors="pt"
                    ).to(self.device)
                except Exception as e:
                    print(f"Warning: multimodal failed, fallback to text: {e}")
                    try:
                        message = [{"role": "user", "content": vera_prompt}]
                        text_result = self.tokenizer.apply_chat_template(
                            message, 
                            add_generation_prompt=True, 
                            tokenize=False
                        )
                        if isinstance(text_result, str):
                            inputs = self.tokenizer(
                                text_result,
                                return_tensors="pt",
                                add_special_tokens=False
                            ).to(self.device)
                        else:
                            formatted_text = f"<|im_start|>user\n{vera_prompt}<|im_end|>\n<|im_start|>assistant\n"
                            inputs = self.tokenizer(
                                formatted_text,
                                return_tensors="pt",
                                add_special_tokens=False
                            ).to(self.device)
                    except Exception as e2:
                        formatted_text = f"<|im_start|>user\n{vera_prompt}<|im_end|>\n<|im_start|>assistant\n"
                        inputs = self.tokenizer(
                            formatted_text,
                            return_tensors="pt",
                            add_special_tokens=False
                        ).to(self.device)
            else:
                message = [{"role": "user", "content": vera_prompt}]
                try:
                    inputs = self.tokenizer.apply_chat_template(
                        message, 
                        add_generation_prompt=True, 
                        return_dict=True, 
                        return_tensors="pt"
                    ).to(self.device)
                except Exception as e:
                    formatted_text = f"<|im_start|>user\n{vera_prompt}<|im_end|>\n<|im_start|>assistant\n"
                    inputs = self.tokenizer(
                        formatted_text,
                        return_tensors="pt",
                        add_special_tokens=False
                    ).to(self.device)
            outputs = self.model.generate(**inputs, max_new_tokens=512, do_sample=False)
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            verifications[vera_type] = self.extract_verifier_approval(response)

        return verifications

    def extract_verifier_approval(self, verifier_response):
        '''
        Extract verifier approval from verifier response.

        Args:
            verifier_response: str, verifier response

        Returns:
            verifier_approval: bool, verifier approval
        '''
        vera_answer_symbol = VERA_ANSWER_SYMBOL.lower()
        pattern = re.compile(
            r'.*{}(.*)'.format(re.escape(vera_answer_symbol)), 
            flags=re.DOTALL | re.IGNORECASE
        )
        match = pattern.search(verifier_response)
        answer = match.group(1).strip() if match else None
        if not answer:
            print(colored(f"WARNING in extract_verifier_approval: {answer=} with {type(answer)=}, "
                        f"and full verifier_response (length {len(verifier_response)}): "
                        f"\n{'-' * 30}\n{verifier_response}\n{'-' * 30} (WARNING in extract_verifier_approval)\n", "yellow"))
            return False
    
        answer = answer.replace("*", "")
        answer = answer.strip().lower()

        if "true" in answer:
            return True
        elif "false" in answer:
            return False
        else:
            print(colored(f"NOTICE extract_verifier_approval: {answer=} not 'true'/'false', "
                        f"checking first word...", "magenta"))
            first_word = answer.split()[0]
            if "true" in first_word:
                print(colored(f"\tSuccess. Found 'true' in first_word.lower(): {first_word.lower()}", "magenta"))
                return True
            elif "false" in first_word:
                print(colored(f"\tSuccess. Found 'false' in first_word.lower(): {first_word.lower()}", "magenta"))
                return False
            else:
                print(colored(f"WARNING in extract_verifier_approval: {answer=} with {type(answer)=} is not 'true' or 'false', "
                            f"AND first word does not contain 'true' or 'false. Full verifier_response: "
                            f"\n{'-' * 30}\n{verifier_response}\n{'-' * 30} (WARNING in extract_verifier_approval)\n", "yellow"))
                return False

    def get_reward(self, question, solution, image=None, processor=None, ground_truth=None, has_options=True, optimization_round=None):
        '''
        Get reward from question and solution.

        Args:
            question: str, question
            solution: str, solution
            image: PIL.Image, optional image for multimodal verification
            processor: processor for handling multimodal inputs
            ground_truth: str, optional ground truth answer for ground-truth-based reward
            has_options: bool, whether the question has options (True for multiple choice, False for open-ended)
            optimization_round: int, optimization round number (0 for cold-start using ground-truth, >0 for TTS without ground-truth)
        Returns:
            reward: float, reward value
        '''
        verifications = self.get_verifications(question, solution, image=image, processor=processor)
        reward = 0
        reward_list = self.get_reward_list()
        total = 0
        for verifier_name, verifier_approval in verifications.items():
            total += reward_list[verifier_name]
            if verifier_approval:
                print(force_colored(f"Verifier {verifier_name} approved.", "green", attrs=['bold']))
            else:
                print(force_colored(f"Verifier {verifier_name} disapproved.", "red", attrs=['bold']))
                reward -= reward_list[verifier_name]

        if self.rule_format_string is not None:
            format_approval = self.get_rule_format_verify(solution)
            if format_approval:
                print(force_colored("Rule format approved.", "green", attrs=['bold']))
            else:
                print(force_colored("Rule format disapproved.", "red", attrs=['bold']))
                reward += -2
        use_ground_truth = (optimization_round is not None and optimization_round == 0)
        if self.ground_truth_weight > 0 and use_ground_truth:
            print(force_colored(f"\n{'='*60}", "cyan", attrs=['bold']))
            print(force_colored("COLD-START (Round 0): ground-truth reward", "cyan", attrs=['bold']))
            print(force_colored(f"Ground-Truth Weight: {self.ground_truth_weight}", "cyan"))
            if ground_truth:
                print(force_colored(f"Ground Truth: {ground_truth}", "cyan"))
            print(force_colored(f"{'='*60}\n", "cyan", attrs=['bold']))
            total += self.ground_truth_weight
            if ground_truth is not None:
                is_correct = is_answer_correct(solution, ground_truth, has_options=has_options)
                if is_correct:
                    print(force_colored("Ground-truth (round 0): APPROVED", "green", attrs=['bold']))
                else:
                    print(force_colored("Ground-truth (round 0): DISAPPROVED", "red", attrs=['bold']))
                    reward -= self.ground_truth_weight
            else:
                print(force_colored("Ground-truth (round 0): SKIPPED (no gt)", "yellow"))
        elif optimization_round is not None and optimization_round > 0:
            print(force_colored(f"TTS (Round {optimization_round}): skip ground-truth", "cyan"))
        print(force_colored(f"Model answer: {solution}", "cyan"))
                
        return reward / total if total > 0 else 0.0

    def get_rule_format_verify(self, solution):
        answer_pattern = self.rule_format_string
        matches = list(re.finditer(answer_pattern, solution, re.DOTALL))
        if len(matches) > 0:
            return True
        else:
            return False
        
    def get_reward_list(self):
        reward_list = {
            "reasoning_check": 2,
            "answer_correct": 3, 
            "answer_completeness": 2,
            "understanding_check": 3,
        }
        return reward_list


MMMURewardModel = VLRewardModel


