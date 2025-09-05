PPO Implementation for BipedalWalker-v3
This repository contains a from-scratch implementation of the Proximal Policy Optimization (PPO) algorithm in PyTorch, designed to solve the BipedalWalker-v3 environment from Gymnasium.

Project Overview
The goal of this project is to provide a clear and understandable implementation of PPO, a state-of-the-art reinforcement learning algorithm. The code is structured to be modular and easy to follow, making it a valuable resource for learning about the core components of PPO and how they interact.

Key Features
Actor-Critic Architecture: Uses separate Actor and Critic networks to learn both a policy and a value function.

Continuous Action Spaces: Handles the continuous action space of the BipedalWalker environment by outputting a normal distribution and clipping actions to the valid range.

PPO Clipped Objective: Implements the signature PPO clipped surrogate objective function to ensure stable policy updates.

GPU Support: Automatically detects and utilizes a CUDA-enabled GPU for accelerated training.

File Structure
main.py: The main script that orchestrates the training process. It contains the PPOAgent class, the main training loop, and hyperparameter definitions.

network.py: Defines the Actor and Critic neural network architectures using torch.nn.Module.

memory.py: Contains the Memory class, a simple buffer for storing and retrieving agent experiences.

How to Run
Clone the repository:

Bash

git clone https://github.com/sankalpkalode49/implementating-proximal-policy-optimization-for-bipedal-walker-in-openAI-gym.git
cd implementating proximal policy optimization for bipedal walker in openAI gym
Install dependencies:
It is recommended to use a virtual environment.

Bash

pip install torch torchvision torchaudio gymnasium box2d-py
Note: Ensure you install a version of PyTorch that is compatible with your GPU's CUDA version.

Start training:

Bash

python main.py
The script will start training the agent, and a window will appear showing the BipedalWalker environment.

Hyperparameters
The main PPO hyperparameters are defined at the top of main.py and are set as follows:

total_timesteps: 2,000,000

update_timestep: 2048

lr_actor: 0.0003

lr_critic: 0.001

gamma: 0.99

K_epochs: 80

eps_clip: 0.2
