from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from simplified_tomato_env import SimplifiedTomatoEnv
import numpy as np

# Hàm huấn luyện PPO
def train_ppo(reward_fun="proxy", total_timesteps=100000, model_name="ppo_tomato"):
    # Tạo môi trường với reward_fun (proxy hoặc true)
    env_config = {"reward_fun": reward_fun, "horizon": 100, "dry_distance": 3, "reward_factor": 0.2, "neg_rew": -0.01}
    env = make_vec_env(lambda: SimplifiedTomatoEnv(env_config), n_envs=1)

    # Khởi tạo mô hình PPO
    model = PPO(
        policy="MultiInputPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        verbose=1
    )

    # Huấn luyện mô hình
    model.learn(total_timesteps=total_timesteps)

    # Lưu mô hình
    model.save(model_name)
    print(f"Model saved as {model_name}.zip")
    return model

if __name__ == "__main__":
    # Huấn luyện với proxy reward (dẫn đến Reward Hacking)
    print("Training with proxy reward...")
    train_ppo(reward_fun="proxy", model_name="ppo_tomato_proxy")
    
    # Huấn luyện với true reward (giảm thiểu Reward Hacking)
    print("\nTraining with true reward...")
    train_ppo(reward_fun="true", model_name="ppo_tomato_true")