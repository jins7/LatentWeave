VERA_ANSWER_SYMBOL = "FINAL VERIFICATION ANSWER IS:"


def get_mmmu_solver_prompt(question: str, has_options: bool = True) -> list:
    if has_options:
        format_instruction = (
            "\n\n"
            "REQUIRED FORMAT (MANDATORY):\n"
            "You must answer this multiple-choice question. Think step by step, then at the very end of your response, output:\n"
            "Final Answer: [X]\n"
            "where X is the chosen option label (A, B, C, D, E, etc.). Do not include any text after this line."
        )
    else:
        format_instruction = (
            "\n\n"
            "REQUIRED FORMAT (MANDATORY):\n"
            "You must answer this open-ended question. Think step by step, then at the very end of your response, output:\n"
            "\\boxed{your final answer}\n"
            "where your final answer is the concise answer to the question. Do not include any text after this line."
        )
    full_content = question + format_instruction
    
    messages = [
        {
            "role": "user",
            "content": full_content,
        },
    ]
    
    return messages


def get_mmmu_verifier_prompt(vera_name: str, question: str, solution: str) -> str:
    system_str = (
        "You are a critical verifier tasked with evaluating multiple-choice question solutions. "
        "You will be presented with a question (including options), an image, and a proposed solution. "
        "Your job is to carefully go over and analyze the solution, paying special attention to how it relates to the provided image. Follow the instructions."
    )

    prefix = f"""{system_str}

QUESTION:
{question}

PROPOSED SOLUTION:
{solution}

"""
    
    vera_names_to_prompts = {
        "reasoning_check": (
            f"{prefix}"
            "INSTRUCTIONS:\n"
            "Your task is to verify the logical consistency and correctness of the reasoning process in the PROPOSED SOLUTION.\n"
            "IMPORTANT: Carefully examine the provided image and verify that the reasoning in the PROPOSED SOLUTION accurately reflects and correctly interprets the visual information shown in the image.\n"
            "1. EXTRACT REASONING STEPS: Identify all logical steps, inferences, and reasoning chains in the PROPOSED SOLUTION.\n"
            "2. VERIFY LOGICAL CONSISTENCY: Check if each reasoning step logically follows from the previous steps and the given information (including visual information from the image).\n"
            "3. CHECK FACTUAL ACCURACY: Verify if the facts, concepts, and information used in the reasoning are accurate and relevant to the question. Pay special attention to whether the solution correctly describes or interprets elements visible in the image.\n"
            "4. EVALUATE REASONING QUALITY: Assess whether the reasoning process is sound, coherent, and leads logically toward the final answer. Ensure the reasoning properly incorporates visual information from the image when relevant.\n"
            f"5. VERIFY: If you find any logical errors, factual inaccuracies, inconsistencies in the reasoning, or mismatches between the solution's description and the actual image content, output '{VERA_ANSWER_SYMBOL}False'. If the reasoning is logically sound, consistent, and accurately reflects the image content, output '{VERA_ANSWER_SYMBOL}True'.\n\n"
            "NOTE: Focus on the quality of reasoning, not just the final answer. Check if the solution demonstrates clear thinking, proper use of concepts, logical progression from premises to conclusion, and accurate interpretation of visual information from the image.\n\n"
        ),

        "answer_correct": (
            f"{prefix}"
            "INSTRUCTIONS:\n"
            "Your task is to determine whether the final answer selected in the PROPOSED SOLUTION is correct.\n"
            "For multiple-choice questions, you need to verify if the selected option (A, B, C, or D) is indeed the correct answer to the question.\n"
            "IMPORTANT: Carefully examine the provided image and verify that the selected answer is correct based on both the question text and the visual information shown in the image.\n"
            "Think through the verification process carefully and logically.\n"
            "IMPORTANT RULES:\n"
            "1. Do NOT analyze the reasoning steps or methods used to arrive at the answer. Focus solely on the final answer's correctness.\n"
            "2. Evaluate whether the selected option correctly answers the question based on your knowledge, understanding of the problem, and careful examination of the image content.\n"
            "3. Consider the question type (mathematical, scientific, factual, analytical, etc.) and verify the answer accordingly. When the question involves visual elements, ensure the answer aligns with what is actually shown in the image.\n"
            "4. Your response must strictly follow the required format:\n"
            f"- If the selected answer option is correct (considering both textual and visual information), respond with: '{VERA_ANSWER_SYMBOL}True'.\n"
            f"- If the selected answer option is incorrect or does not match the image content, respond with: '{VERA_ANSWER_SYMBOL}False'.\n"
        ),
        
        "answer_completeness": (
            f"{prefix}"
            "INSTRUCTIONS:\n"
            "Your task is to verify whether the PROPOSED SOLUTION provides a complete and final answer.\n"
            "For multiple-choice questions, the solution must end with a clear final answer in the format 'Final Answer: [A/B/C/D]'.\n"
            "IMPORTANT: When verifying completeness, also consider whether the solution adequately addresses visual elements from the provided image if the question requires image analysis.\n"
            "Follow these rules carefully:\n"
            "1. Check if the solution reaches a clear and definitive final answer that directly addresses the question (including any visual aspects shown in the image).\n"
            "2. The answer must not be left incomplete or ambiguous, such as:\n"
            "   - Ending with vague statements, uncertainty, or conditional answers instead of a definitive choice.\n"
            "   - Missing a conclusion or final statement explicitly stating the final answer (A, B, C, or D).\n"
            "   - Providing reasoning or analysis but not selecting a final option from the given choices.\n"
            "   - Ending with phrases like 'I think', 'maybe', 'possibly', or 'it could be' without a final selection.\n"
            "   - Failing to address visual information from the image when the question clearly requires image analysis.\n"
            "3. If the solution is incomplete or lacks a final answer, immediately stop checking further and respond in the exact format:\n"
            f"   - '{VERA_ANSWER_SYMBOL}False'\n"
            "4. If the solution is complete and provides a final, explicit answer in the correct format, respond in the exact format:\n"
            f"   - '{VERA_ANSWER_SYMBOL}True'\n"
            
            "Examples:\n"
            "Example 1:\n"
            "Solution: ... reasoning ... Final Answer: A\n"
            f"Your response: '{VERA_ANSWER_SYMBOL}True' (The solution provides a final, definitive answer of A.)\n\n"

            "Example 2:\n"
            "Solution: ... reasoning ... Based on the analysis, option B seems most likely.\n"
            f"Your response: '{VERA_ANSWER_SYMBOL}False' (The answer does not use the required format 'Final Answer: [A/B/C/D]'.)\n\n"

            "Example 3:\n"
            "Solution: ... reasoning ... I cannot determine the answer with certainty.\n"
            f"Your response: '{VERA_ANSWER_SYMBOL}False' (The solution lacks a clear, final answer.)\n"
        ),
        
        "understanding_check": (
            f"{prefix}"
            "INSTRUCTIONS:\n"
            "1. PROBLEM INTERPRETATION:\n"
            "   - Assess if the PROPOSED SOLUTION clearly understands the problem statement, including all key concepts, requirements, and constraints.\n"
            "   - Verify that the solution addresses all relevant aspects of the problem without ignoring any critical detail, condition, or option.\n"
            "   - Check if the solution correctly interprets the question type (mathematical, scientific, factual, analytical, conceptual, etc.).\n"
            "   - Carefully examine the provided image and verify that the solution correctly understands any visual elements, diagrams, charts, or visual information relevant to the question.\n"
            "   - Flag if the solution misinterprets, overlooks, or misunderstands the problem's core requirements, scope, context, or visual information from the image.\n\n"

            "2. OPTION CONSIDERATION:\n"
            "   - Verify that the solution properly considers all given options (A, B, C, D) and evaluates them appropriately.\n"
            "   - Ensure that the solution does not ignore or dismiss options without proper consideration.\n"
            "   - Check if the solution understands what each option represents and how they relate to the question and the image content.\n\n"

            "3. ALIGNMENT WITH THE TASK:\n"
            "   - Verify that the solution responds to the specific multiple-choice question and selects from the given options (A, B, C, D).\n"
            "   - Ensure that the solution does not deviate from the problem's context, provides an answer that is not among the given options, or addresses a different question.\n"
            "   - Check if any critical parts of the problem, options, constraints, or visual elements from the image have been misinterpreted or neglected.\n\n"

            "4. TERMINATION PROTOCOL:\n"
            "   - If the solution clearly misinterprets or fails to address the problem correctly (including visual aspects from the image), stop and respond in the exact format:\n"
            f"     - '{VERA_ANSWER_SYMBOL}False'\n"
            "   - If the solution accurately captures the problem statement, understands all options, correctly interprets visual information from the image when relevant, and aligns with the required solution, respond in the exact format:\n"
            f"     - '{VERA_ANSWER_SYMBOL}True'\n"

            "EXAMPLES:\n"
            "[Case 1] Question: What is the primary function of mitochondria in cells?\n"
            "  Options: A) Protein synthesis  B) Energy production  C) DNA replication  D) Waste removal\n"
            "  Solution: Mitochondria are responsible for producing energy through cellular respiration. Final Answer: B\n"
            "  Assessment: The solution correctly understands the question and selects the correct option.\n"
            f"  Result: '{VERA_ANSWER_SYMBOL}True'\n\n"

            "[Case 2] Question: A shop is selling a drink at 1.5 times the original price. If the original price is $10, what is the new price?\n"
            "  Options: A) $11.50  B) $15.00  C) $16.50  D) $20.00\n"
            "  Solution: The new price is 1.15 * $10 = $11.50. Final Answer: A\n"
            "  Assessment: The solution misinterprets the problem by calculating 1.15 times instead of 1.5 times, showing a misunderstanding of the problem.\n"
            f"  Result: '{VERA_ANSWER_SYMBOL}False'\n\n"

            "[Case 3] Question: Which of the following is a characteristic of Renaissance art?\n"
            "  Options: A) Flat, two-dimensional figures  B) Emphasis on perspective and realism  C) Abstract geometric shapes  D) Minimal use of color\n"
            "  Solution: Renaissance art is known for its focus on perspective, realism, and humanism. Final Answer: B\n"
            "  Assessment: The solution correctly understands the question about Renaissance art characteristics.\n"
            f"  Result: '{VERA_ANSWER_SYMBOL}True'\n\n"

            "CRITICAL REQUIREMENTS:\n"
            "- Assess whether the solution addresses all parts of the problem and considers all given options.\n"
            "- Ensure the solution does not deviate from the problem's intent or misunderstand the question type.\n"
            "- When the question involves visual elements, verify that the solution correctly interprets and uses information from the provided image.\n"
            "- Use exact output formats specified, showing no tolerance for misinterpretations."
        ),

        "image_solution_alignment": (
            f"{prefix}"
            "INSTRUCTIONS:\n"
            "Your task is to verify whether the PROPOSED SOLUTION accurately describes, interprets, and aligns with the visual content shown in the provided image.\n"
            "This verifier focuses specifically on the alignment between the textual solution and the visual information in the image.\n\n"
            "1. VISUAL CONTENT ANALYSIS:\n"
            "   - Carefully examine the provided image and identify all visual elements, including objects, text, diagrams, charts, graphs, mathematical expressions, scientific illustrations, or any other visual information.\n"
            "   - Note the spatial relationships, quantities, measurements, labels, annotations, or any other details visible in the image.\n"
            "   - Identify the key visual information that is relevant to answering the question.\n\n"
            "2. SOLUTION DESCRIPTION VERIFICATION:\n"
            "   - Extract all descriptions, references, or interpretations of visual elements mentioned in the PROPOSED SOLUTION.\n"
            "   - Check if the solution accurately describes what is actually shown in the image.\n"
            "   - Verify that any measurements, quantities, labels, or visual features mentioned in the solution match what appears in the image.\n"
            "   - Ensure that the solution does not describe visual elements that are not present in the image, or miss critical visual information that is present.\n\n"
            "3. ALIGNMENT CHECK:\n"
            "   - Compare the solution's description of visual elements with the actual image content.\n"
            "   - Verify that any calculations, measurements, or observations based on the image are accurate and correctly derived from the visual information.\n"
            "   - Check if the solution correctly interprets diagrams, charts, graphs, or other visual representations shown in the image.\n"
            "   - Ensure that the solution's reasoning that relies on visual information is consistent with what is actually visible in the image.\n\n"
            "4. MISMATCH DETECTION:\n"
            "   - Flag any discrepancies between the solution's description and the actual image content, such as:\n"
            "     * Incorrect descriptions of visual elements (wrong objects, colors, shapes, positions, etc.)\n"
            "     * Misread measurements, numbers, or labels from the image\n"
            "     * Incorrect interpretation of diagrams, charts, or graphs\n"
            "     * Missing or ignoring critical visual information that is relevant to the question\n"
            "     * Describing visual elements that do not exist in the image\n"
            "     * Incorrect spatial relationships or arrangements described in the solution\n\n"
            "5. VERIFICATION PROTOCOL:\n"
            "   - If the solution accurately describes, correctly interprets, and properly aligns with the visual content in the image, respond in the exact format:\n"
            f"     - '{VERA_ANSWER_SYMBOL}True'\n"
            "   - If there are any significant misalignments, inaccuracies, or discrepancies between the solution's description and the actual image content, respond in the exact format:\n"
            f"     - '{VERA_ANSWER_SYMBOL}False'\n\n"
            "EXAMPLES:\n"
            "[Case 1] Question: What is the area of the rectangle shown in the image?\n"
            "  Image: Shows a rectangle with length labeled as 5 cm and width labeled as 3 cm.\n"
            "  Solution: The rectangle has a length of 5 cm and width of 3 cm. Area = 5 × 3 = 15 cm². Final Answer: B\n"
            "  Assessment: The solution correctly reads the dimensions from the image and calculates accurately.\n"
            f"  Result: '{VERA_ANSWER_SYMBOL}True'\n\n"
            "[Case 2] Question: What is the area of the rectangle shown in the image?\n"
            "  Image: Shows a rectangle with length labeled as 5 cm and width labeled as 3 cm.\n"
            "  Solution: The rectangle has a length of 6 cm and width of 4 cm. Area = 6 × 4 = 24 cm². Final Answer: C\n"
            "  Assessment: The solution incorrectly reads the dimensions from the image (5 cm read as 6 cm, 3 cm read as 4 cm).\n"
            f"  Result: '{VERA_ANSWER_SYMBOL}False'\n\n"
            "[Case 3] Question: Which graph shows the correct relationship?\n"
            "  Image: Shows a graph with a linear increasing trend.\n"
            "  Solution: The graph shows a decreasing trend, so the answer is option A. Final Answer: A\n"
            "  Assessment: The solution misinterprets the graph trend (describes as decreasing when it is actually increasing).\n"
            f"  Result: '{VERA_ANSWER_SYMBOL}False'\n\n"
            "CRITICAL REQUIREMENTS:\n"
            "- Pay meticulous attention to the visual details in the image.\n"
            "- Compare every visual reference in the solution with the actual image content.\n"
            "- Be strict about accuracy: even small misreadings or misinterpretations should be flagged.\n"
            "- If the question does not require image analysis, focus on whether the solution incorrectly claims to use visual information.\n"
            "- Use exact output formats specified, showing no tolerance for visual misalignments."
        ),

    }
    
    if vera_name not in vera_names_to_prompts:
        raise ValueError(f"Unknown verifier name: {vera_name}. Available: {list(vera_names_to_prompts.keys())}")
    
    return vera_names_to_prompts[vera_name]

