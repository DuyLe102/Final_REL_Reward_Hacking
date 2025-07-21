# Import thư viện PPO từ stable_baselines3
from stable_baselines3 import PPO 
# Hàm tiện ích để tạo vectorized environment (nhiều bản sao môi trường chạy song song)
from stable_baselines3.common.env_util import make_vec_env
# Import môi trường tùy chỉnh SimplifiedTomatoEnv
from simplified_tomato_env import SimplifiedTomatoEnv
import numpy as np
import time

# Hàm huấn luyện PPO
def train_ppo(reward_fun="proxy", total_timesteps=100000, model_name="ppo_tomato"):
    # Cấu hình cho môi trường huấn luyện (thay đổi phần thưởng bằng cách chọn reward_fun: "proxy" hoặc "true")
    env_config = {"reward_fun": reward_fun,# Hàm phần thưởng: proxy (dễ bị reward hacking) hoặc true (phản ánh đúng mục tiêu)
                   "horizon": 100,# Số bước tối đa trong một episode
                    "dry_distance": 3,# Khoảng cách để xác định cà chua bị khô
                    "reward_factor": 0.2, # Thưởng cho mỗi cà chua được tưới
                    "neg_rew": -0.01} # Phạt nhỏ cho mỗi bước
    env = make_vec_env(lambda: SimplifiedTomatoEnv(env_config), n_envs=1)

    # Khởi tạo mô hình PPO
    model = PPO(
        policy="MultiInputPolicy", # Dùng policy mạng neural xử lý nhiều loại input (obs dạng dict)
        env=env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        verbose=1 # In log huấn luyện ra màn hình
    )

    # Huấn luyện mô hình
    model.learn(total_timesteps=total_timesteps)

    # Lưu mô hình
    model.save(model_name)
    print(f"Model saved as {model_name}.zip")
    return model


if __name__ == "__main__":
    # Huấn luyện với proxy reward (dẫn đến Reward Hacking)
    start = time.time()
    print("Training with proxy reward...")
    train_ppo(reward_fun="proxy", model_name="ppo_tomato_proxy")
    print(f"Proxy training took {time.time() - start:.2f} seconds.\n")

    # Huấn luyện với true reward (giảm thiểu Reward Hacking)
    start = time.time()
    print("Training with true reward...")
    train_ppo(reward_fun="true", model_name="ppo_tomato_true")
    print(f"True training took {time.time() - start:.2f} seconds.")