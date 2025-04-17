import gym
from gym import spaces
import numpy as np
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy
from torch import nn
import torch
from stable_baselines3.common.vec_env import DummyVecEnv

class PixelEnv(gym.Env):
    def __init__(self, features_block, next_true_states, valid_transitions_dict):
        super().__init__()

        self.features = features_block  # shape: (B, F, T)
        self.current_states = features_block[-1, -1]  # état courant: (B,) → dernier feature dernier timestep
        self.true_next_states = next_true_states        # shape: (B,)
        self.valid_transitions = valid_transitions_dict
        self.batch_size = self.features.shape[0]

        # Transitions valides pixel par pixel
        self.possible_next_states = valid_transitions_dict[int(self.current_states)]

        # Action = un entier par pixel
        self.action_space = self.action_space = spaces.Discrete(len(self.possible_next_states))

        # Observation = (B, F+1, T)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.features.shape),
            dtype=np.float32
        )

        self.done = False

    def reset(self):
        self.done = False

        return self.features
    
    def step(self, action):
        if self.done:
            raise Exception("reset() avant de recommencer")

        predicted = self.possible_next_states[action]

        rewards = float(predicted == self.true_next_states)
        self.done = True
        return self.features, rewards, self.done, {}
    
class CustomFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, model, features_dim) :
        super().__init__(observation_space, features_dim)
        self.model = model

    def forward(self, observations):
        return self.model(observations)

class CustomPolicy(ActorCriticPolicy):
    def __init__(self, *args, model, features_dim, **kwargs):
        super().__init__(
            *args,
            features_extractor_class=CustomFeatureExtractor,
            features_extractor_kwargs={'model' : model, 'features_dim':features_dim},
            **kwargs
        )

def create_vec_env_from_block(features_block, next_states, valid_transitions):
    """
    Crée un VecEnv à partir d’un bloc (B, F, T)
    """
    B = features_block.shape[0]
    envs = []

    for i in range(B):
        envs.append(lambda i=i: PixelEnv(
            features_block[i],
            next_true_states=next_states[i],
            valid_transitions_dict=valid_transitions
        ))

    return DummyVecEnv(envs)
