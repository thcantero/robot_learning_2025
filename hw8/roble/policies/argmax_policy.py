import numpy as np



class ArgMaxPolicy(object):

    def __init__(self, critic, use_boltzmann=False):
        self.critic = critic
        self.use_boltzmann = use_boltzmann

    def set_critic(self, critic):
        self.critic = critic

    def get_action(self, obs):
        # MJ: changed the dimension check to a 3
        if len(obs.shape) > 3:
            observation = obs
        else:
            observation = obs[None]

        #debugging
        #print("[argmax_policy] observation shape:", observation.shape)

        # Get Q-values from critic
        q_values = self.critic.qa_values(observation)

        if self.use_boltzmann:
            # Boltzmann exploration (softmax)
            exp_values = np.exp(q_values - np.max(q_values, axis=1, keepdims=True))
            probs = exp_values / np.sum(exp_values, axis=1, keepdims=True)
            action = self.sample_discrete(probs)

        else:
            # Standard argmax exploitation
            action = np.argmax(q_values, axis=1)

        # Remove batch dimension if needed
        if obs.shape == observation.shape[1:]:
            return int(action.squeeze(0))
        return int(action)

    def sample_discrete(self, p):
        # https://stackoverflow.com/questions/40474436/how-to-apply-numpy-random-choice-to-a-matrix-of-probability-values-vectorized-s
        c = p.cumsum(axis=1)
        u = np.random.rand(len(c), 1)
        choices = (u < c).argmax(axis=1)
        return choices

    ####################################
    ####################################