#!/usr/bin/env python3

import os
import argparse
import numpy as np
from copy import copy, deepcopy

import torch
import xml.etree.ElementTree as ET
import gymnasium as gym

from multiprocessing import Value, Array

from stable_baselines3 import PPO
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.env_util import make_vec_env

class Client:

    def __init__(self, args, seed, xml_filepath: str = None):

        self.env_time_horizon = args.n_steps_per_env
        self.rng = np.random.default_rng(seed=seed).uniform # callable random generator

        if args.evn_name.lower() == 'ant':

            # generate xml with random parameters 
            client_custom_xml = None # TODO: load the file
            
            """
            The reward func for ant is four terms, however 'healthy' reward should not change:
            Defaults:
              forward_reward_weight = 1.0
              cntrl_cost_weight = 0.5
              contact_cost_weight = 5e-4
            """

            if args.env_goal == 'random_direction':

                # paper specifies either forward or backwards
                decision = self.rng(low=0.0, high=1.0)
                direction = -1.0 if decision < 0.5 else 1.0

                # init env with "similar" objective function
                self.env = gym.make("Ant-v5",
                                    forward_reward_weight=direction,
                                    #ctrl_cost_weight=self.rng(low=0.3, high=0.7),
                                    #contact_cost_weight=self.rng(low=1e-4, high=9e-4),
                                    #, client_custom_xml) # TODO
                                    )
                
            else:
                # TODO: modify random velocity reward
                raise NotImplementedError
            


        elif args.evn_name.lower() == 'halfcheetah':
            # generate xml with random parameters 
            client_custom_xml = None # TODO: load the file

            """
            The reward func for half cheetah is just two terms
            Defaults:
               forward_reward_weight: float = 1.0,
               ctrl_cost_weight: float = 0.1
            """

            if args.env_goal == 'random_direction':

                # paper specifies either forward or backwards
                decision = self.rng(low=0.0, high=1.0)
                direction = -1.0 if decision < 0.5 else 1.0

                # init env with "similar" objective function
                self.env = gym.make("HalfCheetah-v5",
                                    forward_reward_weight=direction,  
                                    #ctrl_cost_weight=self.rng(low=0.01, high=0.2),
                                    #, client_custom_xml) # TODO
                                    )
                
            else:

                raise NotImplementedError

        policy_kwargs = dict(optimizer_class=torch.optim.SGD, optimizer_kwargs=dict(lr=0.001, momentum=0.9))
        self.model = PPO(
                policy='MlpPolicy',
                env=self.env,
                device='cuda' if args.use_cuda and torch.cuda.is_available() else 'cpu',
                gamma = args.gamma,
                gae_lambda = args.gae_lambda,
                seed = args.seed,
                n_steps = args.n_steps_per_env,
                n_epochs = args.n_epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                ent_coef=args.entropy_coeff,
                verbose=1,
                vf_coef=0.5,
                policy_kwargs=policy_kwargs
                )
        
    def get_trajectory(self) -> RolloutBuffer:

        self.model.rollout_buffer.reset()
        self.model.collect_rollouts(env=self.env, 
                                    callback=None, # TODO
                                    rollout_buffer=self.model.rollout_buffer, 
                                    n_rollout_steps=self.env_time_horizon
                                    )
        
        return self.model.rollout_buffer

################################################################################
class FMRL:
    """ 
    Federated Meta Reinforcement Learning class
    
    original paper: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10339821
    """
    def __init__(self, args):

        # deconstruct args
        self.n_aggregation_rounds: int = args.n_aggregation_rounds
        self.n_local_steps: int = args.n_local_steps

        self.vectorized: bool = args.vectorize_envs # TODO

        self.client_sample_size = int(args.client_sample_coeff * args.n_client_envs)

        self.model_difference_list = []

        self.uniform_rng: callable = np.random.default_rng(seed=args.seed).uniform

        # init client list from args
        self.build_client_list(args) # [(env, index), (env, index), ...]
        
        # env with standard model params and objective functions
        if args.evn_name.lower() == 'ant':
            self.env = gym.make("Ant-v5")

        elif args.evn_name.lower() == 'halfcheetah':
            self.env = gym.make("HalfCheetah-v5")

        
        self.global_model = PPO(
                policy='MlpPolicy',
                env=self.env,
                device='cuda' if args.use_cuda and torch.cuda.is_available() else 'cpu',
                gamma = args.gamma,
                gae_lambda = args.gae_lambda,
                seed = args.seed,
                n_steps = args.n_steps_per_env,
                n_epochs = args.n_epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                ent_coef=args.entropy_coeff,
                verbose=1,
                vf_coef=0.5,
                )   

        # from paper
        self.B1 = 0.9
        self.B2 = 0.999
        self.prev_theta = torch.zeros_like(self.global_model.policy.parameters_to_vector())   
        self.prev_zed = torch.zeros_like(self.global_model.policy.parameters_to_vector())
        self._n = 0.001
        self._k = 1e-8
        

    def build_client_list(self, args):

        self.client_list = []

        # multi thread
        if args.vectorize_envs:
            raise NotImplementedError

        # main thread
        else:
            client_seeds = self.uniform_rng(low=0, high=1000)
            for i in range(args.n_client_envs):
                self.client_list.append(Client(args=args, seed=client_seeds[i]))

    def sample_client_list(self) -> list:
        """
        Construct a random list of Client class objects from main set
        """

        client_sample_list = []

        sample_indices = self.uniform_rng(
                low=0, high=len(self.client_list), size=self.client_sample_size
                )

        # multi thread
        if args.vectorize_envs:
            raise NotImplementedError

        # main thread
        else:
            for idx in sample_indices:
                client = self.client_list[idx]
                client_sample_list.append(client)

        return client_sample_list
    
    def distribute_global_model(self, clients: list[Client]) -> None:
        """
        Update clients with current time global model params
        
        NOTE: This may be confused with load_global_policy() 
        """
        for client in clients:
            client.model.set_parameters(self.global_model.get_parameters())

    def load_global_policy(self, client: Client) -> None:
        """
        Update clients with current time global policy
        
        NOTE: This may be confused with distribute_global_model() 
        """
        # TODO
    
    def collect_trajectory_batch(self, clients: list[Client]) -> list:
        
        trajectory_batch = []

        for client in clients:
            rollout_buffer = client.get_trajectory()
            trajectory_batch.append(rollout_buffer) # TODO: verify rolloutbuffer contains proper data

        return trajectory_batch
    
    def inner_adaption(self, client: Client, rollout: RolloutBuffer) -> None: # TODO
        """
        conduct local model updates
        """
        # alpha = 0.1

        raise NotImplementedError
        return # TODO
    
    def local_step(self, client: Client, rollout: RolloutBuffer) -> None: # TODO
        """
        Perform a single gradient step on a client model
        """
        # client.model.learn()... or client.model.train()...
        raise NotImplementedError
        return  # TODO
    
    def compute_model_delta(self, model_difference_list: list):

        model_delta = self.B1 * \
            (self.prev_theta + (1/len(self.client_list))*torch.sum(model_difference_list)) # TODO: verify
        self.prev_theta = deepcopy(model_delta)

        return model_delta
    
    def compute_zed(self, model_delta):

        zed = (self.B2 * self.prev_zed) + (1 - self.B2)*model_delta**2
        self.prev_zed = deepcopy(zed)

        return zed

    def update_global_model(self, model_delta, zed):
        
        theta_k = self.global_model.policy.parameters_to_vector()

        theta_k1 = theta_k + self._n*(model_delta/(torch.sqrt(zed) * self._k)) # TODO: verify

        self.global_model.policy.load_from_vector(theta_k1)
   

    def train(self):
        """
        Outer loop aggregation rounds
        """
        for k in self.n_aggregation_rounds:

            client_samples: list[Client] = self.sample_client_list()

            self.distribute_global_model(clients=client_samples) # TODO: [3]

            model_diffs = []

            for client in client_samples: # [4-15]

                self.load_global_policy(client=client) # TODO

                rollout_buffer_theta = client.get_trajectory()   

                self.inner_adaption(client=client, rollout=rollout_buffer_theta) # TODO

                rollout_buffer_psi = client.get_trajectory()    

                for t in self.n_local_steps: # [9-12]
                    
                    self.local_step(client=client, rollout=rollout_buffer_psi) 

                    # TODO: [11] update model using eq. (21)

                model_delta_k = None # TODO: [13]

                model_diffs.append(model_delta_k) 

            model_delta = self.compute_model_delta(model_diffs) # TODO [16]

            zed = self.compute_zed(model_delta=model_delta)

            self.update_global_model(model_delta, zed) # TODO [17]

        return self.global_model

################################################################################

def add_args():
    """
    Parse command-line arguments for training configuration.
    """
    parser = argparse.ArgumentParser()

    # Training hyperparameters
    parser.add_argument('--learning_rate', type=float, default=3e-4, 
                                help='Learning rate for the optimizer')
    parser.add_argument('--batch_size', type=int, default=64, 
                                help='Mini-batch size for updates, probably dont change')
    parser.add_argument('--n_epochs', type=int, default=5, 
                                help='Number of epochs to run for each update')
    parser.add_argument('--gamma', type=float, default=0.99, 
                                help='Discount factor for rewards')
    parser.add_argument('--gae_lambda', type=float, default=0.95, 
                                help='Discount factor for rewards')
    parser.add_argument('--entropy_coeff', type=float, default=0.0, 
                                help='Exploration encoragement')

    # Environment and model configuration
    parser.add_argument('--env_name', type=str, choices=['Ant', 'HalfCheetah'], 
                                help='Specify env')
    parser.add_argument('--env_goal', type=str, choices=['random_direction', 'random_velocity'], 
                                help='Specify objective')
    
    # Misc.
    # parser.add_argument('--video_folder', type=str, default='./videos/',
    #                             help='Directory to save recorded videos')
    # parser.add_argument('--video_freq', type=int, default=20000,
    #                             help='Frequency (in timesteps) to record evaluation videos')
    # parser.add_argument('--n_eval_episodes', type=int, default=1,
    #                             help='Number of episodes to record during each evaluation')
    # parser.add_argument('--record_video', action='store_true',
    #                             help='Enable video recording during evaluations')
    parser.add_argument('--env_seed', type=int, default=42, 
                                help='Random seed for the environment')
    parser.add_argument('--model_save_path', type=str, default='./models/', 
                                help='Directory to save trained models')
    parser.add_argument('--tensorboard_log', type=str, default='./tensorboard/', 
                                help='Directory for TensorBoard logs')
    parser.add_argument('--eval_freq', type=int, default=20000, 
                                help='Frequency (in timesteps) to evaluate the agent')
    parser.add_argument('--checkpoint_freq', type=int, default=1, 
                                help='Frequency (in iterations) to save checkpoints')
    parser.add_argument('--use_cuda', action='store_true', 
                                help='Use CUDA for training if available')
    parser.add_argument('--seed', type=int, default=42, 
                                help='Random seed for reproducibility')
    parser.add_argument('--vectorize_envs', type=bool, default=False,
                                help='distribute envs across threads')
    
    
    # FRML parameters
    parser.add_argument('--n_client_envs', type=int, default=200, 
                                help='Number of clinets (independent environments)')
    
    parser.add_argument('--client_sample_coeff', type=int, default=0.2, 
                                help='coeff multiplied by n_client_envs to get sub set selection')
    
    parser.add_argument('--n_steps_per_env', type=int, default=100, 
                                help='Number of steps each environment will take when collectin a \
                                      rollout trajectory, corresponds to H (time horizon) in pseudo\
                                      code')
    
    parser.add_argument('--n_aggregation_rounds', type=int, default=500, 
                                help="Number of outter loop training iterations, corresponds to 'K'\
                                      in pseudo code")
    
    parser.add_argument('--n_local_steps', type=int, default=5, 
                                help='Number of steps per iteration, corresponds to T in pseudo \
                                      code')
    

    args = parser.parse_args()
    return args


if __name__ == "__main__":

    args = add_args()

    fmrl = FMRL(args)

    trained_model = fmrl.train()

    trained_model.save(args.model_save_path)

    
