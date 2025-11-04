import os
import folder_paths
import comfy.sd
import torch
import math
import hashlib
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    Qwen2_5_VLConfig,
    AutoProcessor,
    BitsAndBytesConfig
)
import comfy.utils
from safetensors.torch import load_file
import numpy as np
from PIL import Image
import tempfile, shutil
from huggingface_hub import snapshot_download
import torch.nn.functional as F

# --- Helper Functions ---

def _log(msg: str):
    try:
        print(msg, flush=True)
    except Exception:
        pass

def tensor_to_pil(t):
    if t.ndim == 4:  # (B, C, H, W)
        t = t[0]
    elif t.ndim == 5:  # (1, 1, H, W, C)
        t = t[0, 0]
    if t.shape[0] in (1, 3):  # (C, H, W)
        t = t.permute(1, 2, 0)
    arr = (t.clamp(0, 1).numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr)

def normalize_pil_images(input_imgs):
    """
    Converts a list of ComfyUI image tensors or PIL images to RGB PIL format.
    Matches the format expected by Qwen2.5-VL's processor (like process_vision_info output).
    """
    def tensor_to_pil(t):
        if isinstance(t, torch.Tensor):
            if t.ndim == 4:  # (B, C, H, W)
                t = t[0]
            elif t.ndim == 5:  # (1, 1, H, W, C)
                t = t[0, 0]
            if t.shape[0] in (1, 3):  # (C, H, W)
                t = t.permute(1, 2, 0)
            arr = (t.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
            return Image.fromarray(arr)
        return t  # Already PIL

    def to_rgb(pil_image: Image.Image) -> Image.Image:
        if pil_image.mode == 'RGBA':
            white_background = Image.new("RGB", pil_image.size, (255, 255, 255))
            white_background.paste(pil_image, mask=pil_image.split()[3])
            return white_background
        else:
            return pil_image.convert("RGB")

    result = []
    for img in input_imgs:
        if img is None:
            continue
        pil_img = tensor_to_pil(img)
        if isinstance(pil_img, Image.Image):
            result.append(to_rgb(pil_img))
    return result

class TextEncodeDreamOmni2:
    """
    Produces:
      - enhanced prompt (STRING)
      - list of image tensors (IMAGE_LIST) in ComfyUI tensor format [B,H,W,C]
      - fused multimodal embedding (TENSOR) from Qwen2.5-VL (final hidden states)
    This ensures we carry both the text and the VLM fused embedding downstream.
    """

    _cached_model = None
    _cached_processor = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "mode": (["generate", "edit"],),
                "image_1": ("IMAGE",)
            },
            "optional": {
                "image_2": ("IMAGE",),
                "image_3": ("IMAGE",),
                "image_4": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "encode_dreamomni2"
    CATEGORY = "DreamOmni2"

    def _load_model_once(self):
        if self._cached_model is not None:
            return self._cached_model, self._cached_processor

        model_path = os.path.join(folder_paths.models_dir, "dreamomni2")
        hf_repo = "vantagewithai/DreamOmni2-nf4"

        # Check if local model exists
        if not os.path.exists(model_path) or not os.listdir(model_path):
            _log(f"⚠️ Model not found locally at: {model_path}")
            _log(f"⬇️ Downloading model from Hugging Face repo: {hf_repo}")

            tmp_dir = tempfile.mkdtemp(prefix="dreamomni2_tmp_")

            try:
                snapshot_download(
                    repo_id=hf_repo,
                    local_dir=tmp_dir,
                    local_dir_use_symlinks=False,
                )

                # Move to final location safely
                os.makedirs(model_path, exist_ok=True)
                for item in os.listdir(tmp_dir):
                    shutil.move(os.path.join(tmp_dir, item), model_path)

                _log("✅ Model successfully downloaded and stored locally.")

            except Exception as e:
                _log(f"❌ Download failed: {e}")
                raise e
            finally:
                # Clean up temporary folder if it still exists
                if os.path.exists(tmp_dir):
                    shutil.rmtree(tmp_dir, ignore_errors=True)
        else:
            _log(f"🔹 Using local DreamOmni2 model from: {model_path}")

        # Load processor + model
        processor = AutoProcessor.from_pretrained(model_path)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            device_map="auto",
            dtype=torch.float16,  # modern replacement for torch_dtype
        )

        model.eval()
        _log("✅ DreamOmni2 NF4 quantized model ready.")
        self._cached_model = model
        self._cached_processor = processor
        return model, processor

    def encode_dreamomni2(self, prompt, mode, image_1=None, image_2=None, image_3=None, image_4=None):
        vlm_model, processor = self._load_model_once()

        prefix = " It is generation task." if mode == "generate" else " It is editing task."

        # Collect only images that are actually linked (non-None)
        input_imgs = []
        for img in [image_1, image_2, image_3, image_4]:
            if img is not None:
                input_imgs.append(img)

        # Prepare messages (only include image slots that exist)
        tp = [{"type": "image", "image": f"image_{i+1}"} for i in range(len(input_imgs))]
        tp.append({"type": "text", "text": prompt + prefix})
        messages = [{"role": "user", "content": tp}]
        _log(f"[TextEncodeDreamOmni2] messages: {messages}")

        # Normalize only provided images
        image_inputs = normalize_pil_images(input_imgs)
        inputs = processor(
            text=[processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)],
            images=image_inputs,
            videos=None,
            padding=True,
            return_tensors="pt"
        ).to(vlm_model.device)

        generated_ids = vlm_model.generate(**inputs, do_sample=False, max_new_tokens=4096)
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)

        # Extract the generated content
        def extract_gen_content(text):
            try:
                return text[6:-7]
            except Exception:
                return text

        enhanced_prompt = extract_gen_content(output_text[0]) if output_text else (prompt + prefix)
        return (enhanced_prompt,)

class VantageAdaptiveImageGrid:
    """
    Takes a list of differently sized images and creates an adaptive grid.
    Use CombineImagesList before this node to avoid forced resizing.

    match_image_size = False → uniform grid layout (equal cell sizes)
    match_image_size = True  → adaptive flow (aligns by average row height)
    canvas_width / canvas_height → defines final grid size (keeps aspect ratio per image, no cropping)
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image1": ("IMAGE",),
                "match_image_size": ("BOOLEAN", {"default": True}),
                "spacing_width": (
                    "INT",
                    {"default": 0, "min": 0, "max": 1024, "step": 2},
                ),
                "spacing_color": (
                    ["white", "black", "red", "green", "blue"],
                    {"default": "white"},
                ),
            },
            "optional": {
                "image2": ("IMAGE",),
                "image_3": ("IMAGE",),
                "image_4": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "stitch"
    CATEGORY = "vantage/tools"
            
    def stitch_images(
        self,
        image1,
        direction,
        match_image_size,
        spacing_width,
        spacing_color,
        image2=None,
    ):
        if image2 is None:
            return (image1,)
        # Handle batch size differences
        if image1.shape[0] != image2.shape[0]:
            max_batch = max(image1.shape[0], image2.shape[0])
            if image1.shape[0] < max_batch:
                image1 = torch.cat(
                    [image1, image1[-1:].repeat(max_batch - image1.shape[0], 1, 1, 1)]
                )
            if image2.shape[0] < max_batch:
                image2 = torch.cat(
                    [image2, image2[-1:].repeat(max_batch - image2.shape[0], 1, 1, 1)]
                )

        # Match image sizes if requested
        if match_image_size:
            h1, w1 = image1.shape[1:3]
            h2, w2 = image2.shape[1:3]
            aspect_ratio = w2 / h2

            if direction in ["left", "right"]:
                target_h, target_w = h1, int(h1 * aspect_ratio)
            else:  # up, down
                target_w, target_h = w1, int(w1 / aspect_ratio)

            image2 = comfy.utils.common_upscale(
                image2.movedim(-1, 1), target_w, target_h, "lanczos", "disabled"
            ).movedim(1, -1)

        color_map = {
            "white": 1.0,
            "black": 0.0,
            "red": (1.0, 0.0, 0.0),
            "green": (0.0, 1.0, 0.0),
            "blue": (0.0, 0.0, 1.0),
        }

        color_val = color_map[spacing_color]

        # When not matching sizes, pad to align non-concat dimensions
        if not match_image_size:
            h1, w1 = image1.shape[1:3]
            h2, w2 = image2.shape[1:3]
            pad_value = 0.0
            if not isinstance(color_val, tuple):
                pad_value = color_val

            if direction in ["left", "right"]:
                # For horizontal concat, pad heights to match
                if h1 != h2:
                    target_h = max(h1, h2)
                    if h1 < target_h:
                        pad_h = target_h - h1
                        pad_top, pad_bottom = pad_h // 2, pad_h - pad_h // 2
                        image1 = torch.nn.functional.pad(image1, (0, 0, 0, 0, pad_top, pad_bottom), mode='constant', value=pad_value)
                    if h2 < target_h:
                        pad_h = target_h - h2
                        pad_top, pad_bottom = pad_h // 2, pad_h - pad_h // 2
                        image2 = torch.nn.functional.pad(image2, (0, 0, 0, 0, pad_top, pad_bottom), mode='constant', value=pad_value)
            else:  # up, down
                # For vertical concat, pad widths to match
                if w1 != w2:
                    target_w = max(w1, w2)
                    if w1 < target_w:
                        pad_w = target_w - w1
                        pad_left, pad_right = pad_w // 2, pad_w - pad_w // 2
                        image1 = torch.nn.functional.pad(image1, (0, 0, pad_left, pad_right), mode='constant', value=pad_value)
                    if w2 < target_w:
                        pad_w = target_w - w2
                        pad_left, pad_right = pad_w // 2, pad_w - pad_w // 2
                        image2 = torch.nn.functional.pad(image2, (0, 0, pad_left, pad_right), mode='constant', value=pad_value)

        # Ensure same number of channels
        if image1.shape[-1] != image2.shape[-1]:
            max_channels = max(image1.shape[-1], image2.shape[-1])
            if image1.shape[-1] < max_channels:
                image1 = torch.cat(
                    [
                        image1,
                        torch.ones(
                            *image1.shape[:-1],
                            max_channels - image1.shape[-1],
                            device=image1.device,
                        ),
                    ],
                    dim=-1,
                )
            if image2.shape[-1] < max_channels:
                image2 = torch.cat(
                    [
                        image2,
                        torch.ones(
                            *image2.shape[:-1],
                            max_channels - image2.shape[-1],
                            device=image2.device,
                        ),
                    ],
                    dim=-1,
                )

        # Add spacing if specified
        if spacing_width > 0:
            spacing_width = spacing_width + (spacing_width % 2)  # Ensure even

            if direction in ["left", "right"]:
                spacing_shape = (
                    image1.shape[0],
                    max(image1.shape[1], image2.shape[1]),
                    spacing_width,
                    image1.shape[-1],
                )
            else:
                spacing_shape = (
                    image1.shape[0],
                    spacing_width,
                    max(image1.shape[2], image2.shape[2]),
                    image1.shape[-1],
                )

            spacing = torch.full(spacing_shape, 0.0, device=image1.device)
            if isinstance(color_val, tuple):
                for i, c in enumerate(color_val):
                    if i < spacing.shape[-1]:
                        spacing[..., i] = c
                if spacing.shape[-1] == 4:  # Add alpha
                    spacing[..., 3] = 1.0
            else:
                spacing[..., : min(3, spacing.shape[-1])] = color_val
                if spacing.shape[-1] == 4:
                    spacing[..., 3] = 1.0

        # Concatenate images
        images = [image2, image1] if direction in ["left", "up"] else [image1, image2]
        if spacing_width > 0:
            images.insert(1, spacing)

        concat_dim = 2 if direction in ["left", "right"] else 1
        return (torch.cat(images, dim=concat_dim),)
        
    def stitch(self, image1, match_image_size, spacing_width, spacing_color, image2=None, image_3=None, image_4=None):
        images = [image1, image2, image_3, image_4]
        available = [img for img in images if img is not None]

        if len(available) == 0:
            raise ValueError("At least image_1 must be provided.")
        if len(available) == 1:
            return (available[0],)

        # unpack torch tensor instead of returning nested tuple
        final_img, = self.stitch_images(available[0], "right", match_image_size, spacing_width, spacing_color, available[1])

        if len(available) == 3:
            next_img, = self.stitch_images(final_img, "down", match_image_size, spacing_width, spacing_color, available[2])
            final_img = next_img

        if len(available) == 4:
            final_img1, = self.stitch_images(available[2], "right", match_image_size, spacing_width, spacing_color, available[3])
            final_img2, = self.stitch_images(final_img, "down", match_image_size, spacing_width, spacing_color, final_img1)
            final_img = final_img2

        return (final_img,)


