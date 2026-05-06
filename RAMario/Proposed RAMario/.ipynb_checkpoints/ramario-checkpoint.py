import gym
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
# Define hyperparameters
total_episodes = 1000
max_steps_per_episode = 10000
num_inner_iterations = 5  # Number of inner loop iterations for Reptile

def make_env():
    env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0")
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    return env


# Create the Super Mario Bros environment
env = DummyVecEnv([make_env])
env = VecFrameStack(env, n_stack=4)  # Stack 4 consecutive frames as input

# Initialize the base model
base_model = PPO('CnnPolicy', env)

# Training loop
for episode in range(total_episodes):
    # Perform task-specific training using PPO rollouts per inner iteration
    print(f"Episode {episode + 1}/{total_episodes}")
    for _ in range(num_inner_iterations):
        print(f"  Inner iteration {_ + 1}/{num_inner_iterations}")
        base_model.learn(total_timesteps=max_steps_per_episode, reset_num_timesteps=False)

    # Save the base model's weights after each episode
    base_model.save(f'reptile_model_{episode}.zip')

# Close the environment
env.close()
