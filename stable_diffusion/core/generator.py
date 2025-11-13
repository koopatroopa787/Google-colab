"""
Stable Diffusion Image Generator

This module provides a simple interface for text-to-image generation using
Stable Diffusion models from HuggingFace.
"""

import torch
from diffusers import StableDiffusionXLPipeline
from typing import List, Optional
import time


class StableDiffusionGenerator:
    """
    Wrapper class for Stable Diffusion image generation

    Features:
    - Text-to-image generation
    - Configurable parameters (guidance scale, steps, etc.)
    - Progress tracking
    - Multiple image generation
    """

    def __init__(self, model_name="segmind/SSD-1B", device=None):
        """
        Initialize the Stable Diffusion pipeline

        Args:
            model_name: HuggingFace model identifier
            device: Device to run on (cuda/cpu)
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.pipeline = None
        print(f"Using device: {self.device}")

    def load_model(self, progress_callback=None):
        """
        Load the Stable Diffusion model

        Args:
            progress_callback: Optional callback for progress updates
        """
        if progress_callback:
            progress_callback("Loading model...")

        print(f"Loading {self.model_name}...")

        self.pipeline = StableDiffusionXLPipeline.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            use_safetensors=True,
            variant="fp16" if self.device == "cuda" else None
        )

        self.pipeline.to(self.device)

        if progress_callback:
            progress_callback("Model loaded successfully!")

        print("✓ Model loaded successfully!")

    def generate_image(
        self,
        prompt: str,
        negative_prompt: str = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        num_images: int = 1,
        width: int = 1024,
        height: int = 1024,
        seed: Optional[int] = None,
        progress_callback=None
    ):
        """
        Generate image(s) from text prompt

        Args:
            prompt: Text description of desired image
            negative_prompt: What to avoid in the image
            num_inference_steps: Number of denoising steps (more = better quality, slower)
            guidance_scale: How closely to follow the prompt (higher = more adherence)
            num_images: Number of images to generate
            width: Image width
            height: Image height
            seed: Random seed for reproducibility
            progress_callback: Optional callback for progress updates

        Returns:
            List of generated PIL Images
        """
        if self.pipeline is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        if progress_callback:
            progress_callback("Generating image...")

        # Set seed for reproducibility
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        else:
            generator = None

        # Default negative prompt if not provided
        if negative_prompt is None:
            negative_prompt = (
                "ugly, poorly rendered face, low resolution, poorly drawn feet, "
                "poorly drawn face, out of frame, extra limbs, disfigured, deformed, "
                "body out of frame, blurry, bad composition, blurred, watermark, "
                "grainy, signature, cut off, mutation"
            )

        start_time = time.time()

        # Generate images
        output = self.pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            num_images_per_prompt=num_images,
            width=width,
            height=height,
            generator=generator
        )

        elapsed_time = time.time() - start_time

        if progress_callback:
            progress_callback(f"Generated {num_images} image(s) in {elapsed_time:.2f}s")

        print(f"✓ Generated {num_images} image(s) in {elapsed_time:.2f}s")

        return output.images

    def get_model_info(self):
        """
        Get information about the loaded model

        Returns:
            Dictionary with model information
        """
        if self.pipeline is None:
            return {"status": "Model not loaded"}

        return {
            "model_name": self.model_name,
            "device": str(self.device),
            "dtype": str(self.pipeline.unet.dtype),
            "status": "Ready"
        }


class ImageGenerationPresets:
    """
    Preset configurations for different use cases
    """

    FAST = {
        "num_inference_steps": 25,
        "guidance_scale": 7.0,
        "description": "Fast generation with decent quality"
    }

    BALANCED = {
        "num_inference_steps": 50,
        "guidance_scale": 7.5,
        "description": "Balanced speed and quality"
    }

    QUALITY = {
        "num_inference_steps": 100,
        "guidance_scale": 9.0,
        "description": "High quality, slower generation"
    }

    CREATIVE = {
        "num_inference_steps": 50,
        "guidance_scale": 5.0,
        "description": "More creative, less strict adherence to prompt"
    }

    @classmethod
    def get_preset(cls, name: str):
        """Get preset by name"""
        presets = {
            "Fast": cls.FAST,
            "Balanced": cls.BALANCED,
            "Quality": cls.QUALITY,
            "Creative": cls.CREATIVE
        }
        return presets.get(name, cls.BALANCED)
