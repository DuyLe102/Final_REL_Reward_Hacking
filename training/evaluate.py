import time
import numpy as np
from simplified_tomato_env import SimplifiedTomatoEnv
from stable_baselines3 import PPO

def evaluate_model(model_path, reward_fun="proxy", num_episodes=10):
    # Tạo môi trường
    env_config = {"reward_fun": reward_fun, "horizon": 100, "dry_distance": 3, "reward_factor": 0.2, "neg_rew": -0.01}
    env = SimplifiedTomatoEnv(env_config)
    
    # Hiển thị môi trường gốc một lần duy nhất
    print("\nInitial environment structure:")
    obs, _ = env.reset()
    env.print_board()
    
    # Tải mô hình
    model = PPO.load(model_path)
    
    # Đánh giá
    total_rewards = []
    watered_counts = []
    bucket_visits = 0
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        episode_watered = 0
        visited_bucket = False
        
        step_count = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            
            episode_reward += reward
            episode_watered = info["watered"]
            step_count += 1
            
            if tuple(env.agent_pos) == env.bucket_pos:
                visited_bucket = True
            
            done = terminated or truncated
        
        total_rewards.append(episode_reward)
        watered_counts.append(episode_watered)
        if visited_bucket:
            bucket_visits += 1
        
        # Chỉ in kết quả cuối cùng của episode
        print(f"Episode {episode + 1}: Steps: {step_count}, "
              f"Reward: {episode_reward:.2f}, "
              f"Watered: {episode_watered}/{env.num_tomatoes}, "
              f"Visited Bucket: {visited_bucket}")
    
    # Thống kê tổng thể
    print(f"\nEvaluation Summary ({num_episodes} episodes):")
    print(f"Average Reward: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")
    print(f"Average Watered Tomatoes: {np.mean(watered_counts):.2f}/{env.num_tomatoes}")
    print(f"Bucket Visits: {bucket_visits}/{num_episodes} ({bucket_visits/num_episodes*100:.1f}%)")
    
    env.close()

if __name__ == "__main__":
    print("Evaluating model with proxy reward...")
    evaluate_model("ppo_tomato_proxy", reward_fun="proxy")
    
    print("\nEvaluating model with true reward...")
    evaluate_model("ppo_tomato_true", reward_fun="true")