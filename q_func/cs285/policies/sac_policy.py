from pickle import NONE
from tkinter import TRUE
from cs285.policies.MLP_policy import MLPPolicy
import torch
import numpy as np
from cs285.infrastructure import sac_utils
from cs285.infrastructure import pytorch_util as ptu
from torch import nn
from torch import optim
import itertools

class MLPPolicySAC(MLPPolicy):
    def __init__(self,
                 ac_dim,
                 ob_dim,
                 n_layers,
                 size,
                 discrete=False,
                 learning_rate=3e-4,
                 training=True,
                 log_std_bounds=[-20,2],
                 action_range=[-1,1],
                 init_temperature=1.0,
                 **kwargs
                 ):
        super(MLPPolicySAC, self).__init__(ac_dim, ob_dim, n_layers, size, discrete, learning_rate, training, **kwargs)
        self.log_std_bounds = log_std_bounds
        self.action_range = action_range
        self.init_temperature = init_temperature
        self.learning_rate = learning_rate

        self.log_alpha = torch.tensor(np.log(self.init_temperature)).to(ptu.device)
        self.log_alpha.requires_grad = True
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.learning_rate)

        self.target_entropy = -ac_dim

    @property
    def alpha(self):
        # TODO: Formulate entropy term
        entropy = torch.exp(self.log_alpha)
        return entropy

    def get_action(self, obs: np.ndarray, sample=True) -> np.ndarray:
        # TODO: return sample from distribution if sampling
        # if not sampling return the mean of the distribution

        """Baihe Huang 10/17/2022"""
        
        if len(obs.shape) > 1:
            obs = obs
        else:
            obs = obs[None]

        action_dist = self.forward(ptu.from_numpy(obs))
        if sample:
            actions = action_dist.sample()[0]

            return ptu.to_numpy(actions)
        else:
            return ptu.to_numpy(action_dist.mean)
        """Baihe Huang 10/17/2022"""


    def get_action_torch(self, obs: np.ndarray) -> torch.tensor: #np.ndarray:
        # TODO: return sample from distribution if sampling
        # if not sampling return the mean of the distribution
        
        """Baihe Huang 10/17/2022"""
        
        if len(obs.shape) > 1:
            obs = obs
        else:
            obs = obs[None]

        action_dist = self.forward(ptu.from_numpy(obs))

        actions = action_dist.rsample()

        ac_log_prob = action_dist.log_prob(actions)
        
        return actions, ac_log_prob

        """Baihe Huang 10/17/2022"""


    

    # This function defines the forward pass of the network.
    # You can return anything you want, but you should be able to differentiate
    # through it. For example, you can return a torch.FloatTensor. You can also
    # return more flexible objects, such as a
    # `torch.distributions.Distribution` object. It's up to you!
    def forward(self, observation: torch.FloatTensor):
        # TODO: Implement pass through network, computing logprobs and apply correction for Tanh squashing

        # HINT:
        # You will need to clip log values
        # You will need SquashedNormal from sac_utils file
        
        
        """Baihe Huang 10/17/2022"""

        if self.discrete:
            logits = self.logits_na(observation)
            action_distribution = distributions.Categorical(logits=logits)
            return action_distribution
        else:
            batch_mean = self.mean_net(observation)
            # print(self.logstd, self.log_std_bounds)
            clipped_std = self.logstd.clip(min = self.log_std_bounds[0], max = self.log_std_bounds[1])
            scale_tril = torch.diag(torch.exp(clipped_std))
            batch_dim = batch_mean.shape[0]
            batch_scale_tril = scale_tril.repeat(batch_dim, 1, 1)

            action_distribution = sac_utils.SquashedNormal(
                batch_mean,
                batch_scale_tril
            )
            
            """Baihe Huang 10/17/2022"""

            return action_distribution


    def update(self, ac_log_prob, critic = None):
        # TODO Update actor network and entropy regularizer
        # return losses and alpha value
        # obs = ptu.from_numpy(obs)

        """Baihe Huang 10/17/2022"""

        actor_loss = torch.mean(self.alpha.detach() * ac_log_prob - critic)
        self.optimizer.zero_grad()
        actor_loss.backward()
        self.optimizer.step()


        alpha_loss =  torch.mean(- self.alpha * ac_log_prob.detach()) - self.alpha * self.target_entropy

        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        return actor_loss, alpha_loss, self.alpha
        """Baihe Huang 10/17/2022"""
