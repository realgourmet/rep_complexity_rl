from collections import OrderedDict

from cs285.critics.bootstrapped_continuous_critic import \
    BootstrappedContinuousCritic
from cs285.infrastructure.replay_buffer import ReplayBuffer
from cs285.infrastructure.utils import *
from cs285.infrastructure.sac_utils import *
from cs285.policies.MLP_policy import MLPPolicyAC
from .base_agent import BaseAgent
import gym
import torch
from cs285.policies.sac_policy import MLPPolicySAC
from cs285.critics.sac_critic import SACCritic
import cs285.infrastructure.pytorch_util as ptu

class SACAgent(BaseAgent):
    def __init__(self, env: gym.Env, agent_params):
        super(SACAgent, self).__init__()

        self.env = env
        self.action_range = [
            float(self.env.action_space.low.min()),
            float(self.env.action_space.high.max())
        ]
        self.agent_params = agent_params
        self.gamma = self.agent_params['gamma']
        self.critic_tau = 0.005
        self.learning_rate = self.agent_params['learning_rate']

        self.actor = MLPPolicySAC(
            self.agent_params['ac_dim'],
            self.agent_params['ob_dim'],
            self.agent_params['n_layers'],
            self.agent_params['size'],
            self.agent_params['discrete'],
            self.agent_params['learning_rate'],
            action_range=self.action_range,
            init_temperature=self.agent_params['init_temperature']
        )
        self.actor_update_frequency = self.agent_params['actor_update_frequency']
        self.critic_target_update_frequency = self.agent_params['critic_target_update_frequency']

        self.critic = SACCritic(self.agent_params)
        self.critic_target = copy.deepcopy(self.critic).to(ptu.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.training_step = 0
        self.replay_buffer = ReplayBuffer(max_size=100000)

    def update_critic(self, ob_no, ac_na, next_ob_no, re_n, terminal_n):
        # TODO:
        # 1. Compute the target Q value.
        # HINT: You need to use the entropy term (alpha)
        # 2. Get current Q estimates and calculate critic loss
        # 3. Optimize the critic
        
        
        """Baihe Huang 10/17/2022"""
        q1_pred, q2_pred = self.critic.forward(ob_no, ptu.from_numpy(ac_na))

        next_ac_na, next_log_pi = self.actor.get_action_torch(next_ob_no)

        q1_target, q2_target = self.critic_target.forward(next_ob_no, next_ac_na)

        q_min = torch.min(q1_target.detach(), q2_target.detach()).squeeze(1)

        entropy_reg = next_log_pi * self.actor.alpha.detach()

        target_q_values = ptu.to_numpy(q_min - entropy_reg)

        q_target = ptu.from_numpy(re_n + (1 - terminal_n) * self.gamma * target_q_values).unsqueeze(1)
        
        q_loss = self.critic.loss(q1_pred, q_target.detach()) + self.critic.loss(q2_pred, q_target.detach())

        self.critic.optimizer.zero_grad()
        q_loss.backward()
        self.critic.optimizer.step()
        """Baihe Huang 10/17/2022"""

        return q_loss

    def train(self, ob_no, ac_na, re_n, next_ob_no, terminal_n):
        # TODO
        # 1. Implement the following pseudocode:
        # for agent_params['num_critic_updates_per_agent_update'] steps,
        #     update the critic

        # 2. Softly update the target every critic_target_update_frequency (HINT: look at sac_utils)

        # 3. Implement following pseudocode:
        # If you need to update actor
        # for agent_params['num_actor_updates_per_agent_update'] steps,
        #     update the actor

        # 4. gather losses for logging
        
        """Baihe Huang 10/17/2022"""
        for itr in range(self.agent_params['num_critic_updates_per_agent_update']):

            critic_l = self.update_critic(ob_no, ac_na, next_ob_no, re_n, terminal_n)
                
        soft_update_params(self.critic, self.critic_target, self.critic_tau)

        for _ in range(self.agent_params['num_actor_updates_per_agent_update']):


            ac_na, ac_log_prob = self.actor.get_action_torch(ob_no)

            q1_target, q2_target = self.critic_target.forward(ob_no, ac_na)

            q_critic = torch.min(q1_target, q2_target).squeeze(1)

            actor_l, alpha_l, alpha = self.actor.update(ac_log_prob, q_critic)


        loss = OrderedDict()
        loss['Critic_Loss'] = critic_l
        loss['Actor_Loss'] = actor_l
        loss['Alpha_Loss'] = alpha_l
        loss['Temperature'] = alpha
        """Baihe Huang 10/17/2022"""


        return loss

    def add_to_replay_buffer(self, paths):
        self.replay_buffer.add_rollouts(paths)

    def sample(self, batch_size):
        # return self.replay_buffer.sample_recent_data(batch_size)
        return self.replay_buffer.sample_random_data(batch_size)

    def save_actor(self, filepath):
        torch.save(self.actor.state_dict(), filepath)

    def save_critic(self, filepath):
        torch.save(self.critic.Q1.state_dict(), filepath)
