"""
BipedalWalker RL Training Interface

Train an AI agent to walk using reinforcement learning (PPO algorithm).
"""

import gradio as gr
from core.trainer import BipedalWalkerTrainer
import matplotlib.pyplot as plt


# Global trainer instance
trainer = None


def create_trainer(n_envs, n_steps, batch_size):
    """Create a new trainer"""
    global trainer

    try:
        trainer = BipedalWalkerTrainer(
            n_envs=int(n_envs),
            n_steps=int(n_steps),
            batch_size=int(batch_size)
        )

        trainer.create_environment()
        trainer.create_model()

        return "✓ Trainer created successfully!"
    except Exception as e:
        return f"❌ Error: {str(e)}"


def train_agent(timesteps, progress=gr.Progress()):
    """Train the agent"""
    global trainer

    if trainer is None:
        return "Please create a trainer first!"

    try:
        progress(0, desc="Training...")

        trainer.train(total_timesteps=int(timesteps))

        progress(0.8, desc="Evaluating...")
        mean_reward, std_reward = trainer.evaluate()

        progress(1.0, desc="Complete!")

        return f"✓ Training complete!\n\nEvaluation: {mean_reward:.2f} +/- {std_reward:.2f}"
    except Exception as e:
        return f"❌ Error: {str(e)}"


def evaluate_agent():
    """Evaluate the trained agent"""
    global trainer

    if trainer is None:
        return "Please create and train a model first!"

    try:
        mean_reward, std_reward = trainer.evaluate()
        return f"Mean Reward: {mean_reward:.2f} +/- {std_reward:.2f}"
    except Exception as e:
        return f"Error: {str(e)}"


def save_model_fn(path):
    """Save the model"""
    global trainer

    if trainer is None:
        return "No model to save!"

    try:
        trainer.save_model(path)
        return f"✓ Model saved to {path}"
    except Exception as e:
        return f"Error: {str(e)}"


def load_model_fn(path):
    """Load a model"""
    global trainer

    if trainer is None:
        trainer = BipedalWalkerTrainer()

    try:
        trainer.load_model(path)
        return "✓ Model loaded successfully!"
    except Exception as e:
        return f"Error: {str(e)}"


def get_env_info_fn():
    """Get environment information"""
    try:
        trainer_temp = BipedalWalkerTrainer()
        info = trainer_temp.get_env_info()

        return f"""
**Environment Information:**

- **Observation Space**: {info['observation_space_shape']}
- **Action Space**: {info['action_space_shape']}

**Sample Observation**: {info['observation_space_sample'][:5]}... (24 values total)

**Sample Action**: {info['action_space_sample']}
"""
    except Exception as e:
        return f"Error: {str(e)}"


def create_interface():
    """Create Gradio interface"""

    with gr.Blocks(title="BipedalWalker RL", theme=gr.themes.Soft()) as app:
        gr.Markdown("""
        # 🤸 BipedalWalker: Reinforcement Learning Training

        Train an AI agent to walk using **PPO** (Proximal Policy Optimization).

        **The Challenge**: Control a 2D bipedal robot to walk forward without falling.

        **Observation**: 24 values (hull angle, angular velocity, joint angles, leg contact, etc.)

        **Action**: 4 continuous values (torque for hip/knee joints)

        **Reward**: Moving forward earns positive reward, falling gives -100
        """)

        with gr.Tabs():
            with gr.Tab("🚀 Training"):
                gr.Markdown("### Configure and Train Agent")

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("**Hyperparameters**")

                        n_envs_slider = gr.Slider(
                            1, 32,
                            value=16,
                            step=1,
                            label="Parallel Environments"
                        )

                        n_steps_slider = gr.Slider(
                            512, 2048,
                            value=1024,
                            step=256,
                            label="Steps per Update"
                        )

                        batch_size_slider = gr.Slider(
                            32, 256,
                            value=64,
                            step=32,
                            label="Batch Size"
                        )

                        create_btn = gr.Button("Create Trainer", variant="primary")
                        create_status = gr.Textbox(label="Status", interactive=False)

                        create_btn.click(
                            create_trainer,
                            inputs=[n_envs_slider, n_steps_slider, batch_size_slider],
                            outputs=create_status
                        )

                    with gr.Column():
                        gr.Markdown("**Training**")

                        timesteps_slider = gr.Slider(
                            100_000, 10_000_000,
                            value=1_000_000,
                            step=100_000,
                            label="Total Timesteps"
                        )

                        train_btn = gr.Button("🏋️ Start Training", variant="primary", size="lg")
                        train_status = gr.Textbox(label="Training Status", lines=5, interactive=False)

                        train_btn.click(
                            train_agent,
                            inputs=timesteps_slider,
                            outputs=train_status
                        )

            with gr.Tab("📊 Evaluation"):
                gr.Markdown("### Evaluate Trained Agent")

                eval_btn = gr.Button("Evaluate Agent", variant="primary")
                eval_output = gr.Textbox(label="Evaluation Results", lines=3)

                eval_btn.click(evaluate_agent, outputs=eval_output)

            with gr.Tab("💾 Save/Load"):
                gr.Markdown("### Save or Load Model")

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("**Save Model**")
                        save_path_input = gr.Textbox(
                            label="Save Path",
                            value="ppo_bipedalwalker.zip"
                        )
                        save_btn = gr.Button("Save Model")
                        save_status = gr.Textbox(label="Status", interactive=False)

                        save_btn.click(
                            save_model_fn,
                            inputs=save_path_input,
                            outputs=save_status
                        )

                    with gr.Column():
                        gr.Markdown("**Load Model**")
                        load_path_input = gr.Textbox(
                            label="Load Path",
                            value="ppo_bipedalwalker.zip"
                        )
                        load_btn = gr.Button("Load Model")
                        load_status = gr.Textbox(label="Status", interactive=False)

                        load_btn.click(
                            load_model_fn,
                            inputs=load_path_input,
                            outputs=load_status
                        )

            with gr.Tab("ℹ️ About"):
                gr.Markdown("""
                ## About BipedalWalker & PPO

                ### 🎯 Environment: BipedalWalker-v3

                A 2D physics simulation where you control a bipedal robot:
                - **Goal**: Walk forward as far as possible
                - **Observations**: 24 continuous values (angles, velocities, contacts)
                - **Actions**: 4 continuous values (joint torques)
                - **Reward**: +1 for each frame moving forward, -100 for falling

                ### 🧠 Algorithm: PPO (Proximal Policy Optimization)

                **Why PPO?**
                - More stable than vanilla policy gradients
                - Prevents large policy updates that can break learning
                - Works well with continuous action spaces
                - Industry standard for many RL tasks

                **How it works:**
                1. Collect experience using current policy
                2. Calculate advantages (how much better than expected)
                3. Update policy with clipped objective
                4. Repeat

                **Key Hyperparameters:**
                - **n_envs**: More environments = more diverse experience
                - **n_steps**: Steps collected before each update
                - **batch_size**: Training batch size
                - **gamma**: Discount factor (future reward importance)
                - **gae_lambda**: Advantage estimation smoothing

                ### 📈 Training Tips:

                1. **Start small**: Try 100K-1M timesteps first
                2. **Be patient**: Good walking typically appears after 1-5M steps
                3. **Parallel environments**: More = faster but needs more memory
                4. **Evaluation**: Run evaluation to get true performance

                ### 🎓 Learning Resources:

                - [OpenAI Spinning Up](https://spinningup.openai.com/)
                - [PPO Paper](https://arxiv.org/abs/1707.06347)
                - [Gymnasium Docs](https://gymnasium.farama.org/)

                ### Expected Performance:

                - **Random policy**: ~-100 (falls immediately)
                - **Decent policy**: 50-150
                - **Good policy**: 200-300+
                """)

                gr.Markdown("### Environment Details")
                info_btn = gr.Button("Show Environment Info")
                info_output = gr.Markdown()

                info_btn.click(get_env_info_fn, outputs=info_output)

    return app


if __name__ == "__main__":
    app = create_interface()
    app.launch(share=True)
