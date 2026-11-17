import os
import pandas as pd
import matplotlib.pyplot as plt
from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer

# 1. Load Dataset
if not os.path.exists("samples.csv"):
    raise FileNotFoundError("samples.csv not found in directory!")

df = pd.read_csv("samples.csv")
dataset = Dataset.from_pandas(df.rename(columns={"prompt_text": "prompt"}))

# 2. Final Balanced Reward Function
def balanced_camel_case_reward(prompts, completions, ground_truth, **kwargs):
    rewards = []
    for comp, truth in zip(completions, ground_truth):
        clean_out = comp.strip()
        truth = truth.strip()
        
        if clean_out == truth:
            rewards.append(1.0)
            continue
            
        score = 0.0
        if "_" not in clean_out and clean_out:
            score += 0.3
        if clean_out and clean_out[0].islower():
            score += 0.2
        if clean_out.isalnum():
            score += 0.2
        if clean_out.lower() == truth.lower():
            score += 0.2
            
        rewards.append(min(score, 0.9))
    return rewards

# 3. GRPO Configuration
training_args = GRPOConfig(
    output_dir="./grpo_camelcase_out",
    learning_rate=5e-6,
    per_device_train_batch_size=4,
    num_generations=4,
    max_completion_length=32,
    num_train_epochs=3,
    logging_steps=1,
    save_strategy="no",
    report_to="none"
)

# 4. Initialize Trainer
trainer = GRPOTrainer(
    model="Qwen/Qwen2.5-0.5B-Instruct",
    reward_funcs=balanced_camel_case_reward,
    args=training_args,
    train_dataset=dataset,
)

print("Starting GRPO training...")
trainer.train()

# 5. Generate and Save Training Curve
history = trainer.state.log_history
steps = [x["step"] for x in history if "reward" in x]
rewards = [x["reward"] for x in history if "reward" in x]

if steps and rewards:
    plt.figure(figsize=(8, 5))
    plt.plot(steps, rewards, marker="o", linestyle="-", color="#2b5c8f", label="Mean Group Reward")
    plt.title("GRPO Training Curve: snake_case to camelCase")
    plt.xlabel("Training Step")
    plt.ylabel("Average Reward (0.0 to 1.0)")
    plt.ylim(-0.05, 1.05)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig("training_curve.png", dpi=300)
    print("Saved training_curve.png successfully!")
