import numpy as np


class ArgMaxPolicy(object):

    def __init__(self, critic):
        self.critic = critic

    def get_action(self, obs):
        if len(obs.shape) > 3:
            observation = obs
        else:
            observation = obs[None]
        
        ## TODO return the action that maxinmizes the Q-value 
        # at the current observation as the output
        """Baihe Huang 10/12/2022"""
        action = np.argmax(self.critic.qa_values(observation))
        """Baihe Huang 10/12/2022"""

        return action.squeeze()