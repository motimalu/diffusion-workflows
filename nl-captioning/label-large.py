import re
import torch
from pathlib import Path
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info
import time

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)

model_id = "Qwen/Qwen3-VL-32B-Instruct"

model = Qwen3VLForConditionalGeneration.from_pretrained(
    model_id,
    quantization_config=quant_config,
    device_map="auto",
    trust_remote_code=True,
    attn_implementation="flash_attention_2"
)

processor = AutoProcessor.from_pretrained(
    model_id,
    min_pixels=256*28*28,
    max_pixels=1024*28*28 
)

input_dir = Path("/mnt/c/data/images/")

extensions = [".jpg", ".jpeg", ".png", ".webp"]

example_tags_single = r"fern (sousou no frieren), sousou no frieren, @van gogh, 1girl, arm at side, black background, black coat, black robe, blue butterfly, blunt bangs, blunt ends, blush, bright pupils, bug, butterfly, butterfly on hand, chromatic aberration, closed mouth, coat, dress, eyelashes, feet out of frame, from above, half updo, hand up, lips, long hair, long sleeves, looking at viewer, mage staff, puffy sleeves, purple butterfly, purple eyes, purple hair, purple pupils, robe, sidelocks, signature, simple background, solo, staff, standing, straight hair, tsurime, upturned eyes, very long hair, white dress, wide sleeves"

example_response_single = r"Artwork of Fern from Sousou no Frieren by @van gogh, with long purple hair styled in a half-updo with blunt bangs and sidelocks, and striking purple eyes with bright pupils. She wears a white dress under a dark, wide-sleeved robe with a black coat draped over her shoulders. Her expression is gentle, with a faint blush on her cheeks and her lips closed, as she looks directly at the viewer. Her right hand is raised, palm up, holding a vibrant purple butterfly. A mage staff is visible behind her, and the scene is set against a simple black background with chromatic aberration effects. The artist's signature 'Vincent' is visible near the butterfly."

example_tags_multiple = r"gawr gura, mori calliope, ninomae ina'nis, takanashi kiara, watson amelia, hololive, @van gogh, 5girls, indoors, on couch, sitting, blonde hair, blue eyes, black shirt, blue hair, shark fin, pink hair, red eyes, purple hair, purple eyes, purple shirt, orange hair, closed eyes, smiling"

example_response_multiple = r"A cozy indoor scene by @van gogh featuring a group of five Hololive girls relaxing on a sofa: Watson Amelia, Gawr Gura, Mori Calliope, Ninomae Ina'nis, and Takanashi Kiara. The blonde-haired girl wears a black shirt and sits on the left. Next to her, the blue-haired girl has shark fins and wears a white shirt. In the center, the pink-haired girl has red eyes and smiles in a black outfit. The purple-haired girl wears a purple shirt with tentacles, while the orange-haired girl has her eyes closed and smiles in an orange outfit. The girls are sitting, lying, or interacting with each other, some with closed eyes, or smiling."


# Detect if more than one character is present based on grounding tags.
def get_is_multiple_characters(tags: str):
    # Match '2girls', '3boys', '5others', etc.
    if re.search(r"\dgirls|\dboys|\dothers", tags):
        return True
    
    # Count occurrences of '1girl', '1boy', or '1other'
    individuals = re.findall(r"1girl|1boy|1other", tags)
    if len(individuals) >= 2:
        return True
        
    return False

# Find the last terminal punctuation (. ! ?) and remove any trailing text that doesn't form a complete sentence.
def trim_incomplete_sentence(text: str) -> str:
    terminators = ('.', '!', '?')
    
    last_idx = max((text.rfind(t) for t in terminators))
    
    if last_idx != -1:
        return text[:last_idx + 1].strip()
    
    return text.strip()

# Format grounding tags before sharing with VLM
def clean_tag_string(tags: str, keep_tags_n: int = 50, exclude_artists = False):
    # Remove artist names (starts with @, followed by word characters)
    if exclude_artists:
        tags = re.sub(r'@\w+,?\s*', '', tags)

    # Remove backslashes (escaping parentheses) - don't want these propagated into the NL captions
    tags = tags.replace('\\', '')

    tags = ", ".join(tags.split(', ')[:keep_tags_n])

    return tags

def process_images():
    for img_file in input_dir.iterdir():
        if img_file.suffix.lower() in extensions:
            start_time = time.perf_counter()
            is_multiple_characters = False
            
            tags_file = img_file.with_suffix(".txt")
            output_file = img_file.parent / f"{img_file.stem}_nl.txt"

            tags_content = ""
            if tags_file.exists():
                with open(tags_file, "r", encoding="utf-8") as f:
                    tags_str = f.read().strip()
                    is_multiple_characters = get_is_multiple_characters(tags_str)
                    tags_content = clean_tag_string(tags_str)

            multiple_characters_prompt = [
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": (
                            "You are an artistic and creative image captioner for an art database. "
                            "Your task is to create short, descriptive captions of images using Image Tags for guidance.\n"
                            "GUIDELINES:\n"
                            "- Focus on the visual description and what is not described in the image by the Image Tags.\n"
                            "- Reference the start of the Image Tags which describe first any characters in the image, followed the copywrights to which they belong, followed by the artist name.\n"
                            "- Artist names are represented with the @ symbol prefix, i.e. @van gogh \n"
                            "- Parenthesis around tags are used as Qualifiers serve to make the exact meaning of the tag clear so there is no confusion with other tags. \n"
                            "- If the same tag is repeated directly after a character name it can be considered to be the copyright. "
                            "i.e. \"fern (sousou no frieren), sousou no frieren\" tags can be interpreted as \"Fern from Sousou no Frieren\". \n"
                            "Qualifiers can also indicate alternate forms or outfits of characters, i.e. \"hoshimachi suisei (micomet), hololive\". \n"
                            "In these cases or if unsure, its fine to describe the character keeping the qualifier in parenthesis, i.e. \"Hoshimachi Suisei (micomet) from Hololive\". \n"
                            "STEPS: \n"
                            "1. List the names and copyrights of all characters present (e.g., 'A cozy indoor scene by @quasarcake featuring a group of five Hololive girls relaxing on a sofa: Watson Amelia, Gawr Gura, Mori Calliope, Ninomae Ina'nis, and Takanashi Kiara.' "
                            "2. Assign each a unique visible anchor (e.g., 'The pink-haired girl' or 'The girl on the left')."
                            "3. Use these anchor for the remainder of the caption to describe individual actions or outfits. Do not repeat names. (e.g., 'On the left, the pink-haired girl has red eyes and smiles in a black outfit.')\n"
                            "EXAMPLE: \n"
                            f"Example Tags Input: {example_tags_multiple}\n"
                            f"Example Caption Output: {example_response_multiple}\n"
                        )}
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": str(img_file.absolute())},
                        {"type": "text", "text": (
                            f"Image Tags: {tags_content}\n\n"
                            "Instruction: Use the above Image Tags as a guideline to provide natural language captions for the image. Avoid complicated formatting like unusual punctuation marks, lists, or markdown."
                        )},
                    ],
                }
            ]

            single_character_prompt = [
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": (
                            "You are an artistic and creative image captioner for an art database. "
                            "Your task is to create short, descriptive captions of images using Image Tags for guidance.\n"
                            "GUIDELINES:\n"
                            "- Focus on the visual description and what is not described in the image by the Image Tags.\n"
                            "- Reference the start of the Image Tags which describe first any characters in the image, followed the copywrights to which they belong, followed by the artist name.\n"
                            "- Artist names are represented with the @ symbol prefix, i.e. @van gogh \n"
                            "- Parenthesis around tags are used as Qualifiers serve to make the exact meaning of the tag clear so there is no confusion with other tags. \n"
                            "- If the same tag is repeated directly after a character name it can be considered to be the copyright. "
                            "i.e. \"fern (sousou no frieren), sousou no frieren\" tags can be interpreted as \"Fern from Sousou no Frieren\". \n"
                            "Qualifiers can also indicate alternate forms or outfits of characters, i.e. \"hoshimachi suisei (micomet), hololive\". \n"
                            "In these cases or if unsure, its fine to describe the character keeping the qualifier in parenthesis, i.e. \"Hoshimachi Suisei (micomet) from Hololive\". \n"
                            "STEPS: \n"
                            "1. List the name, copyright, and artist of the character, followed by their basic appearance (e.g., 'Artwork of Fern from Sousou no Frieren by @van gogh, with long purple hair styled in a half-updo with blunt bangs and sidelocks.')\n"
                            "2. Describe the characters specific outfit, expression, and the background.\n"
                            "3. Describe any other elements not present in the Image Tags, such as text."
                            "EXAMPLE: \n"
                            f"Example Tags Input: {example_tags_single}\n"
                            f"Example Caption Output: {example_response_single}\n"
                        )}
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": str(img_file.absolute())},
                        {"type": "text", "text": (
                            f"Image Tags: {tags_content}\n\n"
                            "Instruction: Use the above Image Tags as a guideline to provide natural language captions for the image. Avoid complicated formatting like unusual punctuation marks, lists, or markdown."
                        )},
                    ],
                }
            ]

            messages = multiple_characters_prompt if is_multiple_characters else single_character_prompt

            image_inputs, video_inputs = process_vision_info(messages)

            inputs = processor(
                text=[processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)],
                images=image_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(model.device)

            generated_ids = model.generate(**inputs, max_new_tokens=480)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )

            trimmed_output = trim_incomplete_sentence(output_text[0])

            with open(output_file, "w", encoding="utf-8") as f:
                f.write(trimmed_output)

            end_time = time.perf_counter()
            elapsed = end_time - start_time
                
            print(f"Processed as {"MUTLIPLE CHARACTERS" if is_multiple_characters else "SINGLE CHARACTER"}: {img_file.name} -> {output_file.name} in {elapsed:.2f} seconds.")


if __name__ == "__main__":
    process_images()
