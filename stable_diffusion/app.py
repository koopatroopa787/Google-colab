"""
Stable Diffusion Web Interface

Interactive text-to-image generation with:
- Multiple presets (Fast, Balanced, Quality, Creative)
- Parameter controls
- Image gallery
- Progress tracking
"""

import gradio as gr
from core.generator import StableDiffusionGenerator, ImageGenerationPresets
import torch


# Global generator instance
generator = None
status_text = "Not initialized"


def initialize_model(progress=gr.Progress()):
    """Initialize the Stable Diffusion model"""
    global generator, status_text

    progress(0, desc="Starting...")

    try:
        generator = StableDiffusionGenerator()
        progress(0.3, desc="Loading model...")
        generator.load_model()
        status_text = "✓ Model loaded and ready!"
        progress(1.0, desc="Ready!")
        return status_text
    except Exception as e:
        status_text = f"❌ Error: {str(e)}"
        return status_text


def generate_images(
    prompt,
    negative_prompt,
    preset,
    num_steps,
    guidance,
    num_images,
    width,
    height,
    seed,
    use_seed,
    progress=gr.Progress()
):
    """Generate images from text prompt"""
    global generator

    if generator is None:
        return None, "❌ Please initialize the model first!"

    try:
        # Use seed if checkbox is checked
        actual_seed = int(seed) if use_seed else None

        progress(0, desc="Generating...")

        images = generator.generate_image(
            prompt=prompt,
            negative_prompt=negative_prompt if negative_prompt else None,
            num_inference_steps=int(num_steps),
            guidance_scale=float(guidance),
            num_images=int(num_images),
            width=int(width),
            height=int(height),
            seed=actual_seed
        )

        progress(1.0, desc="Complete!")

        return images, f"✓ Generated {len(images)} image(s) successfully!"

    except Exception as e:
        return None, f"❌ Error: {str(e)}"


def apply_preset(preset_name):
    """Apply generation preset"""
    preset = ImageGenerationPresets.get_preset(preset_name)
    return preset["num_inference_steps"], preset["guidance_scale"]


def create_interface():
    """Create Gradio interface"""

    with gr.Blocks(title="Stable Diffusion", theme=gr.themes.Soft()) as app:
        gr.Markdown("""
        # 🎨 Stable Diffusion: Text-to-Image Generation

        Create stunning images from text descriptions using Stable Diffusion XL.

        **Features:**
        - Multiple quality presets
        - Adjustable parameters
        - Batch generation
        - Seed control for reproducibility
        """)

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 🚀 Setup")

                init_btn = gr.Button("Initialize Model", variant="primary", size="lg")
                status_output = gr.Textbox(label="Status", value="Not initialized", interactive=False)

                init_btn.click(initialize_model, outputs=status_output)

                gr.Markdown("### 📝 Prompt")

                prompt_input = gr.Textbox(
                    label="Prompt",
                    placeholder="Describe the image you want to generate...",
                    lines=3,
                    value="A beautiful sunset over mountains, highly detailed, 8K, photorealistic"
                )

                negative_prompt_input = gr.Textbox(
                    label="Negative Prompt (optional)",
                    placeholder="What to avoid...",
                    lines=2,
                    value=""
                )

                gr.Markdown("### ⚙️ Settings")

                preset_dropdown = gr.Dropdown(
                    ["Fast", "Balanced", "Quality", "Creative"],
                    label="Preset",
                    value="Balanced"
                )

                with gr.Row():
                    num_steps_slider = gr.Slider(
                        10, 150,
                        value=50,
                        step=5,
                        label="Inference Steps"
                    )
                    guidance_slider = gr.Slider(
                        1, 20,
                        value=7.5,
                        step=0.5,
                        label="Guidance Scale"
                    )

                preset_dropdown.change(
                    apply_preset,
                    inputs=preset_dropdown,
                    outputs=[num_steps_slider, guidance_slider]
                )

                with gr.Row():
                    width_slider = gr.Slider(
                        512, 1024,
                        value=1024,
                        step=64,
                        label="Width"
                    )
                    height_slider = gr.Slider(
                        512, 1024,
                        value=1024,
                        step=64,
                        label="Height"
                    )

                num_images_slider = gr.Slider(
                    1, 4,
                    value=1,
                    step=1,
                    label="Number of Images"
                )

                with gr.Row():
                    use_seed_checkbox = gr.Checkbox(label="Use Seed", value=False)
                    seed_input = gr.Number(label="Seed", value=42, precision=0)

                generate_btn = gr.Button("🎨 Generate Image", variant="primary", size="lg")

            with gr.Column(scale=2):
                gr.Markdown("### 🖼️ Generated Images")

                output_gallery = gr.Gallery(
                    label="Results",
                    show_label=True,
                    columns=2,
                    height=600
                )

                generation_status = gr.Textbox(label="Generation Status", interactive=False)

        generate_btn.click(
            generate_images,
            inputs=[
                prompt_input,
                negative_prompt_input,
                preset_dropdown,
                num_steps_slider,
                guidance_slider,
                num_images_slider,
                width_slider,
                height_slider,
                seed_input,
                use_seed_checkbox
            ],
            outputs=[output_gallery, generation_status]
        )

        gr.Markdown("""
        ---
        ### 💡 Tips for Better Results:

        - **Be specific**: Include details like style, lighting, mood, colors
        - **Use negative prompts**: Specify what you don't want in the image
        - **Adjust guidance**: Higher = more adherence to prompt, lower = more creative
        - **Steps matter**: More steps = better quality but slower
        - **Presets**:
          - **Fast**: Quick results, good for testing
          - **Balanced**: Good quality-speed tradeoff
          - **Quality**: Best results, takes longer
          - **Creative**: More artistic freedom

        ### 🎯 Example Prompts:

        - "A futuristic city at night, neon lights, cyberpunk style, highly detailed, 8K"
        - "Oil painting of a serene lake, mountains in background, sunset, impressionist style"
        - "Portrait of a wise old wizard, fantasy art, detailed face, magical atmosphere"
        - "Modern minimalist interior design, bright natural lighting, Scandinavian style"
        """)

    return app


if __name__ == "__main__":
    app = create_interface()
    app.launch(share=True)
