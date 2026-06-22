import asyncio
import argparse
import base64
import json
import logging
import math
import re
import time

from datetime import datetime
from io import BytesIO
from openai import AsyncOpenAI
from pathlib import Path
from PIL import Image
from typing import TypedDict, Optional, Dict, Any, List

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = f"{timestamp}_bulk-caption-extended.txt"
logging.basicConfig(
    filename=log_filename,
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Defaults for Qwen3.6-35B-A3B inference on RTX 6000 Pro
MODEL_NAME = "Qwen3.6-35B-A3B"
MAX_TOKENS = 81920
TEMPERATURE = 0.6
TOP_P = 0.95
TOP_K = 20
PRESENCE_PENALTY = 0.0
ENABLE_THINKING = True

META_DIR = "/mnt/c/data/0_danbooru_meta/"
INPUT_DIR = "/mnt/c/data/1_collection_name3/"
OUTPUT_FILE_SUFFIX = "_nl"
BATCH_SIZE = 64
LOG_THINKING = False
CAPTION_ONLY_NSFW = False
SKIP_ALREADY_LABELLED = True
LONG_CAPTION = False

logging.info(f"Starting bulk-caption-extended.py with config:\nMODEL_NAME: {MODEL_NAME}\nENABLE_THINKING: {ENABLE_THINKING}\nLOG_THINKING: {LOG_THINKING}\nCAPTION_ONLY_NSFW: {CAPTION_ONLY_NSFW}\nSKIP_ALREADY_LABELLED: {SKIP_ALREADY_LABELLED}\nLONG_CAPTION: {LONG_CAPTION}\nOUTPUT_FILE_SUFFIX: {OUTPUT_FILE_SUFFIX}\nBATCH_SIZE:{BATCH_SIZE}")

client = AsyncOpenAI(
    api_key="EMPTY",
    base_url="http://localhost:8000/v1",
    timeout=3600
)

img_extensions = [".png", ".jpg", ".jpeg", ".webp"]

example_tags_single = r"fern \\(sousou no frieren\\), sousou no frieren, @van gogh, 1girl, arm at side, black background, black coat, black robe, blue butterfly, blunt bangs, blunt ends, blush, bright pupils, bug, butterfly, butterfly on hand, chromatic aberration, closed mouth, coat, dress, eyelashes, feet out of frame, from above, half updo, hand up, lips, long hair, long sleeves, looking at viewer, mage staff, puffy sleeves, purple butterfly, purple eyes, purple hair, purple pupils, robe, sidelocks, signature, simple background, solo, staff, standing, straight hair, tsurime, upturned eyes, very long hair, white dress, wide sleeves"

example_response_single = r"Fern from Sousou no Frieren by @van gogh, with long purple hair styled in a half-updo with blunt bangs and sidelocks, and striking purple eyes with bright pupils. She wears a white dress under a dark, wide-sleeved robe with a black coat draped over her shoulders. Her expression is gentle, with a faint blush on her cheeks and her lips closed, as she looks directly at the viewer. Her right hand is raised, palm up, holding a vibrant purple butterfly. A mage staff is visible behind her, and the scene is set against a simple black background with chromatic aberration effects. The artist's signature 'lze' is visible near the butterfly."

example_tags_multiple = r"gawr gura, mori calliope, ninomae ina'nis, takanashi kiara, watson amelia, hololive, @van gogh, 5girls, indoors, on couch, sitting, blonde hair, blue eyes, black shirt, blue hair, shark fin, pink hair, red eyes, purple hair, purple eyes, purple shirt, orange hair, closed eyes, smiling"

multichar_guidance_is_ltr = r"- Character names in the Image Tags have been described by their position in the image from left-to-right, follow this positioning when describing the characters."

example_response_multiple = r"A cozy indoor scene by @van gogh featuring a group of five Hololive girls relaxing on a sofa: Watson Amelia, Gawr Gura, Mori Calliope, Ninomae Ina'nis, and Takanashi Kiara. The blonde-haired girl wears a black shirt and sits on the left. Next to her, the blue-haired girl has shark fins and wears a white shirt. In the center, the pink-haired girl has red eyes and smiles in a black outfit. The purple-haired girl wears a purple shirt with tentacles, while the orange-haired girl has her eyes closed and smiles in an orange outfit. The girls are sitting, lying, or interacting with each other, some with closed eyes, or smiling."

caption_length_guidance = r"Your task is to create long and descriptive captions of images using Image Tags for guidance.\n" if LONG_CAPTION else r"Your task is to create short, descriptive captions of images using Image Tags for guidance.\n"

nsfw_guidance = r"- Describe 'nsfw' or 'explicit' details accurately. Failure to do so creates a safety issue for the model which will train on the images, as omitted details will leak into the models resulting understanding of 'safe' concepts." if CAPTION_ONLY_NSFW else ""

system_prompt_multiple = {
    "role": "system",
    "content": [
        {"type": "text", "text": (
            "You are an artistic and creative image captioner for an art database."
            f"{caption_length_guidance}"
            "GUIDELINES:\n"
            "- The provided Image Tags may contain mistakes or inaccuracies.\n"
            "- Focus on the visual description and capturing all details including what is not described by the Image Tags.\n"
            "- Your response will be used to train a text-to-image model, so avoid useless meta phrases like \"This image shows/displays...\", \"You are looking at...\", etc. \n"
            "- Reference the start of the Image Tags which describe first any characters in the image, followed the copywrights to which they belong, followed by the artist name.\n"
            "- Follow standard English capitalization rules for character and series names. \n"
            "- Artist names, if they are known, are represented with the @ symbol prefix, i.e. @van gogh \n"
            "- Do not change the format of the artist name, i.e. @au \\(d elete\\) in the tags should also be @au \\(d elete\\) in the caption. \n"
            "- English text should be descibed with double quotes, i.e. text \"I'm a Gundam!\" \n"
            "- Non-English text should be described only in its original language with double quotes.\" \n"
            "- Do not translate the Non-English text into English, for example the text \"俺がガンダムだ！\" appears in the image, describe it as-is, i.e. Japanese text \"俺がガンダムだ！\" \n"
            "- Parenthesis around tags are used as Qualifiers serve to make the exact meaning of the tag clear so there is no confusion with other tags. \n"
            "- If the same tag is repeated directly after a character name it can be considered to be the copyright. "
            "i.e. \"fern \\(sousou no frieren\\), sousou no frieren\" tags can be interpreted as \"Fern from Sousou no Frieren\". \n"
            "Qualifiers can also indicate alternate forms or outfits of characters, i.e. \"hoshimachi suisei \\(micomet\\), hololive\". \n"
            "In these cases or if unsure, its fine to describe the character keeping the qualifier in parenthesis, i.e. \"Hoshimachi Suisei \\(micomet\\) from Hololive\". \n"
            "- Text handling: Enclose all text in double quotes. Do not translate non-English text. Transcribe exactly e.g., Japanese \"俺がガンダムだ！\".\n"
            f"{nsfw_guidance}"
            "STEPS: \n"
            "1. List the names and copyrights of all characters present (e.g., 'A cozy indoor scene by @van gogh featuring a group of five Hololive girls relaxing on a sofa: Watson Amelia, Gawr Gura, Mori Calliope, Ninomae Ina'nis, and Takanashi Kiara.'\n"
            "2. Assign each a unique visible anchor (e.g., 'The pink-haired girl on the right' or 'The blue-eyed girl on the left').\n"
            "3. Use these anchor for the remainder of the caption to describe individual actions or outfits. Do not repeat names. (e.g., 'On the left, the pink-haired girl has red eyes and smiles in a black outfit.')\n"
            "4. Describe the spacial relationships between any other elements in the scene.\n"
            "5. Describe any other elements not present in the Image Tags, such as text.\n"
            "EXAMPLE: \n"
            f"Example Tags Input: {example_tags_multiple}\n"
            f"Example Caption Output: {example_response_multiple}\n"
        )}
    ],
}

system_prompt_single = {
    "role": "system",
    "content": [
        {"type": "text", "text": (
            "You are an artistic and creative image captioner for an art database."
            f"{caption_length_guidance}"
            "GUIDELINES:\n"
            "- The provided Image Tags may contain mistakes or inaccuracies.\n"
            "- Focus on the visual description and capturing all details including what is not described by the Image Tags.\n"
            "- Your response will be used to train a text-to-image model, so avoid useless meta phrases like \"This image shows/displays...\", \"You are looking at...\", etc. \n"
            "- Reference the start of the Image Tags which describe first any characters in the image, followed the copywrights to which they belong, followed by the artist name.\n"
            "- Follow standard English capitalization rules for character and series names. \n"
            "- Artist names, if they are known, are represented with the @ symbol prefix, e.g., @van gogh \n"
            "- Do not change the format of the artist name, e.g., @au \\(d elete\\) in the tags should also be @au \\(d elete\\) in the caption. \n"
            "- Parenthesis around tags are used as Qualifiers serve to make the exact meaning of the tag clear so there is no confusion with other tags. \n"
            "- If the same tag is repeated directly after a character name it can be considered to be the copyright. "
            "i.e. \"fern \\(sousou no frieren\\), sousou no frieren\" tags can be interpreted as \"Fern from Sousou no Frieren\". \n"
            "Qualifiers can also indicate alternate forms or outfits of characters, e.g., \"hoshimachi suisei \\(micomet\\), hololive\". \n"
            "In these cases or if unsure, its fine to describe the character keeping the qualifier in parenthesis, e.g., \"Hoshimachi Suisei \\(micomet\\) from Hololive\". \n"
            "- Text handling: Enclose all text in double quotes. Do not translate non-English text. Transcribe exactly e.g., Japanese \"俺がガンダムだ！\".\n"
            f"{nsfw_guidance}"
            "STEPS: \n"
            "1. List the name, copyright, and artist of the character, followed by their basic appearance (e.g., 'Artwork of Fern from Sousou no Frieren by @van gogh, with long purple hair styled in a half-updo with blunt bangs and sidelocks.')\n"
            "2. Describe the characters specific outfit, expression, and the background.\n"
            "3. Describe spacial relationships between any other elements in the scene.\n"
            "4. Describe any other elements not present in the Image Tags, such as text.\n"
            "EXAMPLE: \n"
            f"Example Tags Input: {example_tags_single}\n"
            f"Example Caption Output: {example_response_single}\n"
        )}
    ],
}

class CaptionItem(TypedDict):
    img_file: Path
    is_ltr: bool
    tags_str: str
    meta_dict: Optional[Dict[str, Any]]

def format_meta_to_json_str(meta_dict: dict) -> str:
    def clean_and_split(tag_string: str, is_artist=False):
        if not tag_string:
            return []
        tags = tag_string.split()
        cleaned = []
        for t in tags:
            t_clean = t.replace('_', ' ')
            if is_artist and not t_clean.startswith('@'):
                t_clean = f"@{t_clean}"
            cleaned.append(t_clean)
        return cleaned

    structured_data = {
        "characters": clean_and_split(meta_dict.get("tag_string_character", "")),
        "copyrights": clean_and_split(meta_dict.get("tag_string_copyright", "")),
        "artists": clean_and_split(meta_dict.get("tag_string_artist", ""), is_artist=True),
        "general_tags": clean_and_split(meta_dict.get("tag_string_general", ""))
    }
    
    return json.dumps(structured_data, ensure_ascii=False, indent=2)

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

# Rely on grounding tags being accurately labeled for nsfw/explicit
def get_is_nsfw(tags: str):
    return bool(re.search(r"nsfw|explicit", tags))

def get_has_leftright(tags: str):
    return bool(re.search(r"character names", tags))

# Detects if an image exceeds size_limit, resizes if necessary, and returns a base64 data URL.
def process_image_for_vlm(img_path: Path, size_limit: int = 1280) -> str:
    with Image.open(img_path) as img:
        width, height = img.size
        area_limit = size_limit ** 2
        current_area = width * height
        
        # If Image is small enough, return the direct file path
        if current_area <= area_limit:
            return f"file://{img_path.absolute()}"
        
        # If Image is too large, calculate new dimensions based on area ratio
        # Ratio 'r' is the square root of (current_area / target_area)
        r = (current_area / area_limit) ** 0.5
        new_width = int(math.ceil(width / r))
        new_height = int(math.ceil(height / r))
        
        # Convert to RGB to ensure compatibility (e.g., stripping Alpha from PNGs/WebPs)
        if img.mode != "RGB":
            img = img.convert("RGB")

        img = img.resize((new_width, new_height), Image.LANCZOS)

        buffered = BytesIO()
        img.save(buffered, format="JPEG", quality=95)
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        return f"data:image/jpeg;base64,{img_str}"

existing_filenames = {p.name for p in INPUT_DIR.rglob("*.txt")}

def make_conversation(image_url: str, caption_item: CaptionItem):
    tags_content = caption_item["tags_str"]
    meta_prompt_str = ""
    if caption_item["meta_dict"]:
        meta_prompt_str = f"Image Metadata: \n{format_meta_to_json_str(caption_item["meta_dict"])}\n Instruction: Use the above Image Metadata as a source of truth for the image."

    ltr_guidance_str = ""
    if caption_item["is_ltr"]:
        ltr_guidance_str = f"Positioning Note: {multichar_guidance_is_ltr}\n"

    return [{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": { "url": image_url}},
            {"type": "text", "text": (
                f"Image Tags: {tags_content}\n\n"
                "Instruction: Use the above Image Tags as a guideline to provide natural language captions for the image. \n"
                "Avoid complicated formatting like unusual punctuation marks, lists, or markdown. \n"
                f"{ltr_guidance_str}"
                f"{meta_prompt_str}"
            )},
        ],
    }]

async def caption_single(caption_item: CaptionItem, is_multiple_characters: bool):
    conversations: list[dict[str, Any]] = []
    image_url = process_image_for_vlm(caption_item["img_file"])
    new_conversation = make_conversation(image_url, caption_item)
    conversations.extend(new_conversation)
    if is_multiple_characters:
        messages_list = [system_prompt_multiple] + conversations
    else:
        messages_list = [system_prompt_multiple] + conversations

    response = await client.chat.completions.create(
        messages=messages_list,
        model=MODEL_NAME,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        presence_penalty=PRESENCE_PENALTY,
        extra_body={
            "top_k": TOP_K,
            "chat_template_kwargs": {"enable_thinking": False} if not ENABLE_THINKING else None
        }, 
    )
    
    refusal = response.choices[0].message.refusal
    if refusal is not None:
        error = f"Model Refusal received for {caption_item['img_file']}: {refusal}"
        logging.error(error)
        print(error)
        return

    text = response.choices[0].message.content

    if text is None:
        error = f"Empty model output received for {caption_item['img_file']}"
        print(error)
        logging.error(error)
        logging.error(response)
        return
    
    img_file = Path(caption_item["img_file"])
    output_file_name = f"{img_file.stem}{OUTPUT_FILE_SUFFIX}.txt"
    output_file = img_file.parent / output_file_name
    with open(output_file, 'w') as f:
        logging.info(f"Saving file: {output_file_name}")
        f.write(text.strip())

async def do_caption(items: list[CaptionItem], batch_type):
    try:
        num_images = len(items)
        for i in list(range(0, num_images, BATCH_SIZE)):
            start_time = time.perf_counter()
            batch = items[i:i+BATCH_SIZE]
            tasks = [caption_single(item, batch_type == multi_type) for item in batch]
            await asyncio.gather(*tasks)
            end_time = time.perf_counter()
            elapsed = end_time - start_time
            message = f"Progress: {i} / {num_images}. {batch_type}: Processed {len(batch)} images in {elapsed:.2f}s."
            logging.info(message)
            print(message)
    except Exception as e:
            end_time = time.perf_counter()
            elapsed = end_time - start_time
            message = f"Exception occurred: {e}"
            logging.info(message)
            print(message)

multi_type = "MULTIPLE CHARACTERS"
single_type = "SINGLE CHARACTER"

async def run_main():
    caption_items_multichar: List[CaptionItem] = []
    caption_items_singlechar: List[CaptionItem] = []

    for img_file in Path(INPUT_DIR).rglob('*'):
        if img_file.suffix.lower() in img_extensions:
            danbooru_id_match = re.search(r"\d+", img_file.stem)
            tags_file = img_file.with_suffix(".txt")
            output_file_name = f"{img_file.stem}{OUTPUT_FILE_SUFFIX}.txt"

            if SKIP_ALREADY_LABELLED and output_file_name in existing_filenames:
                logging.info(f"Skipping {img_file.name} as already labelled.")
                continue
            if tags_file.exists():
                with open(tags_file, "r", encoding="utf-8") as f:
                    tags_str = f.read().strip()
                    is_multiple_characters = get_is_multiple_characters(tags_str)
                    is_ltr = get_has_leftright(tags_str)
                    is_nsfw = get_is_nsfw(tags_str)
                    if (is_nsfw and not CAPTION_ONLY_NSFW) or (not is_nsfw and CAPTION_ONLY_NSFW):
                        logging.info(f"Skipping {img_file.name} as is_nswf: {is_nsfw} when CAPTION_ONLY_NSFW: {CAPTION_ONLY_NSFW}.")
                        continue
            meta_dict = None
            if danbooru_id_match:
                meta_file_path = Path(META_DIR) / f"danbooru_{danbooru_id_match.group()}_meta.json"
                if meta_file_path.exists():
                    try:
                        with open(meta_file_path, "r", encoding="utf-8") as f:
                            meta_dict = json.load(f)
                    except Exception as e:
                        logging.warning(f"Exception loading JSON meta for {img_file.name}: {e}")
                else:
                    logging.warning(f"No JSON meta found for {img_file.name}")

            item: CaptionItem = {
                "img_file": img_file,
                "is_ltr": is_ltr,
                "tags_str": tags_str,
                "meta_dict": meta_dict
            }
            if is_multiple_characters:
                caption_items_multichar.append(item)
            else:
                caption_items_singlechar.append(item)

    await do_caption(caption_items_multichar, multi_type)
    await do_caption(caption_items_singlechar, single_type)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the inference script with customizable parameters.")

    parser.add_argument("--input-dir", type=str, default=INPUT_DIR, help="Input directory path")
    parser.add_argument("--meta-dir", type=str, default=META_DIR, help="Danbooru meta files directory path")
    parser.add_argument("--model-name", type=str, default=MODEL_NAME, help="Model name to use")
    parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS, help="Maximum number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=TEMPERATURE, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=TOP_P, help="Nucleus sampling probability")
    parser.add_argument("--top-k", type=int, default=TOP_K, help="Top-K sampling value")
    parser.add_argument("--presence-penalty", type=float, default=PRESENCE_PENALTY, help="Presence penalty")
    parser.add_argument("--output-file-suffix", type=str, default=OUTPUT_FILE_SUFFIX, help="Suffix for output files")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size for processing")

    parser.add_argument("--enable-thinking", action=argparse.BooleanOptionalAction, default=ENABLE_THINKING, help="Enable or disable thinking")
    parser.add_argument("--log-thinking", action=argparse.BooleanOptionalAction, default=LOG_THINKING, help="Enable or disable logging of thinking")
    parser.add_argument("--caption-only-nsfw", action=argparse.BooleanOptionalAction, default=CAPTION_ONLY_NSFW, help="Only caption NSFW images")
    parser.add_argument("--skip-already-labelled", action=argparse.BooleanOptionalAction, default=SKIP_ALREADY_LABELLED, help="Skip files that are already labelled")
    parser.add_argument("--long-caption", action=argparse.BooleanOptionalAction, default=LONG_CAPTION, help="Enable long captions")

    args = parser.parse_args()

    INPUT_DIR = args.input_dir
    META_DIR = args.meta_dir
    MODEL_NAME = args.model_name
    MAX_TOKENS = args.max_tokens
    TEMPERATURE = args.temperature
    TOP_P = args.top_p
    TOP_K = args.top_k
    PRESENCE_PENALTY = args.presence_penalty
    ENABLE_THINKING = args.enable_thinking
    LOG_THINKING = args.log_thinking
    CAPTION_ONLY_NSFW = args.caption_only_nsfw
    SKIP_ALREADY_LABELLED = args.skip_already_labelled
    LONG_CAPTION = args.long_caption
    OUTPUT_FILE_SUFFIX = args.output_file_suffix
    BATCH_SIZE = args.batch_size

    asyncio.run(run_main())
