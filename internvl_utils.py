import torch
import torchvision.transforms as T
from PIL import Image
import os
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig
import time
import json
import io
import re
from logger import logger
import threading
import os
import math

from dotenv import load_dotenv, set_key

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
model_lock = threading.Lock()

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    if isinstance(image_file, bytes):
        image = Image.open(io.BytesIO(image_file)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

def save_model_env(model_path, quant_mode):
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    set_key(env_path, "INTERNVL_MODEL_PATH", model_path)
    set_key(env_path, "INTERNVL_QUANT_MODE", quant_mode)

def load_model_env():
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    load_dotenv(env_path)
    model_path = os.getenv("INTERNVL_MODEL_PATH")
    quant_mode = os.getenv("INTERNVL_QUANT_MODE", "4bit")
    return model_path, quant_mode

# The 'split_model' function is designed for multi-GPU setups.
# It manually calculates how to distribute model layers across available GPUs.
# For a single-GPU setup, we will not use this function, but it is kept as requested.
def split_model(model_name=None):
    """
    Membagi layer model ke device_map sesuai jumlah GPU.
    Jika model_name None, akan mengambil dari env (load_model_env).
    """
    if model_name is None:
        model_path, _ = load_model_env()
        if model_path is None:
            model_name = "InternVL2_5-4B"
        else:
            # Ambil nama model dari path, fallback ke default jika tidak ditemukan
            model_name = os.path.basename(model_path.rstrip("/"))
            if not model_name:
                model_name = "InternVL2_5-4B"
    # Mapping jumlah layer untuk model yang didukung
    num_layers_map = {
        'InternVL2_5-4B': 36,
        'InternVL2_5-8B': 32
        # Tambahkan mapping lain jika ada model lain
    }
    if model_name not in num_layers_map:
        raise Exception(f"Unknown model_name '{model_name}' for split_model. Please update num_layers_map.")
    num_layers = num_layers_map[model_name]
    device_map = {}
    world_size = torch.cuda.device_count()
    # Since the first GPU will be used for ViT, treat it as half a GPU.
    num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
    num_layers_per_gpu = [num_layers_per_gpu] * world_size
    num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            device_map[f'language_model.model.layers.{layer_cnt}'] = i
            layer_cnt += 1
    device_map['vision_model'] = 0
    device_map['mlp1'] = 0
    device_map['language_model.model.tok_embeddings'] = 0
    device_map['language_model.model.embed_tokens'] = 0
    device_map['language_model.model.rotary_emb'] = 0
    device_map['language_model.output'] = 0
    device_map['language_model.model.norm'] = 0
    device_map['language_model.lm_head'] = 0
    device_map[f'language_model.model.layers.{num_layers - 1}'] = 0

    return device_map

# Load model path and quant_mode from .env if available, else use default
model_path_env, quant_mode_env = load_model_env()
if model_path_env and os.path.isdir(model_path_env):
    path = model_path_env
    quant_mode = quant_mode_env
else:
    path = "OpenGVLab/InternVL2_5-4B"
    quant_mode = "4bit"

# Ambil nama model dari path untuk split_model secara dinamis
model_name_for_split = os.path.basename(path.rstrip("/")) if path else "InternVL2_5-4B"
# The line below, which calls the multi-GPU split_model function, is not needed for single-GPU inference.
# device_map = split_model(model_name_for_split)
model_kwargs = {
    "torch_dtype": torch.bfloat16,
    "low_cpu_mem_usage": True,
    "use_flash_attn": False,
    "trust_remote_code": True,
    # "device_map": "auto" is the standard way to let Hugging Face Transformers
    # handle device placement automatically. On a single-GPU system, it will
    # correctly place the model on 'cuda:0'.
    "device_map": "auto"
}
# The following global model loading block was already commented out in the original code.
# if quant_mode == "4bit":
#     model_kwargs["load_in_4bit"] = True
#     model = AutoModel.from_pretrained(path, **model_kwargs).eval()
# elif quant_mode == "8bit":
#     model_kwargs["load_in_8bit"] = True
#     model = AutoModel.from_pretrained(path, **model_kwargs).eval()
# else:
#     model = AutoModel.from_pretrained(path, **model_kwargs).eval().cuda()
# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=quant_mode == "4bit",
#     load_in_8bit=quant_mode == "8bit"
# )



tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)
generation_config = dict(max_new_tokens=1024, do_sample=False)

def set_internvl_model(model_path, quant_mode="4bit"):
    """
    Change the global InternVL model and tokenizer safely.
    quant_mode: "4bit", "8bit", or "16bit" (default "4bit")
    """
    global model, tokenizer, path
    import torch
    from transformers import AutoModel, AutoTokenizer
    with model_lock:
        # Validasi folder model_path
        if not os.path.isdir(model_path):
            raise Exception(f"Model path '{model_path}' does not exist or is not a directory.")
        # Validasi quant_mode
        valid_quant = {"4bit", "8bit", "16bit"}
        if quant_mode not in valid_quant:
            raise Exception(f"quant_mode must be one of {valid_quant}, got '{quant_mode}'")
        del model
        del tokenizer
        torch.cuda.empty_cache()
        path = model_path
        # Ambil nama model dari path untuk split_model secara dinamis
        model_name_for_split = os.path.basename(model_path.rstrip("/")) if model_path else "InternVL2_5-4B"

        # The following two lines are for a multi-GPU setup.
        # We comment out the call to split_model and instead use device_map = "auto"
        # to let the library handle placement on a single GPU.
        # device_map = split_model(model_name_for_split)
        device_map = "auto"

        model_kwargs = {
            "torch_dtype": torch.bfloat16,
            "low_cpu_mem_usage": True,
            "use_flash_attn": False,
            "trust_remote_code": True,
            "device_map": device_map
        }
        # The following model loading block was already commented out in the original code.
        # if quant_mode == "4bit":
        #     model_kwargs["load_in_4bit"] = True
        #     model = AutoModel.from_pretrained(path, **model_kwargs).eval()
        # elif quant_mode == "8bit":
        #     model_kwargs["load_in_8bit"] = True
        #     model = AutoModel.from_pretrained(path, **model_kwargs).eval()
        # else:
        #     # 16bit: no quantization, move to cuda after eval
        #     model = AutoModel.from_pretrained(path, **model_kwargs).eval().cuda()
        model = AutoModel.from_pretrained(
            path,
            quantization_config=bnb_config if quant_mode in ["4bit", "8bit"] else None,
            torch_dtype=torch.bfloat16,
            device_map=device_map,
            trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)
        logger.info(f"InternVL model changed to {model_path} with quant_mode={quant_mode}")
        save_model_env(model_path, quant_mode)

def ask_internvl_ai(files=None, prompt="", max_num=12):
    import torch
    import time
    pixel_values = None
    with model_lock:
        if files:
            start_time = time.time()
            tensors_list = []
            for image_data in files:
                # load_image processes one image (in bytes) and returns its tensor tiles.
                tensors_list.append(load_image(image_data, max_num=max_num).to(torch.bfloat16))
            if tensors_list:
                pixel_values = torch.cat(tensors_list, dim=0)
                # This line correctly moves the tensor to the model's device,
                # which will be 'cuda:0' in a single-GPU setup.
                pixel_values = pixel_values.to(model.device)

            logger.info(f'Waktu eksekusi load_image: {(time.time() - start_time):.2f} s')
        
        start_time = time.time()
        response = model.chat(tokenizer, pixel_values, prompt, generation_config)
        logger.info(f': {prompt}\nAssistant: {response}')
        logger.info(f'\n\nWaktu eksekusi response: {(time.time() - start_time):.2f} s\n')
        return response

def extract_json_from_response(response: str):
    json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
    if json_match:
        json_content = json_match.group(1)
        try:
            extracted_json = json.loads(json_content)
            logger.info(f"Extracted JSON content: {json.dumps(extracted_json, indent=2)}")
            return extracted_json
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from response: {e}")
            return {}
    else:
        logger.warning("No JSON content found in the response")
        return {}

def add_default_result_fields(result):
    result['pdfExtract'] = True
    result['ocrResult'] = [
        [
            {
                "transcription": " "
            }
        ]
    ]
    return result