from .base_critic import BaseCritic
import torch
import torch.optim as optim
from torch.nn import utils
from torch import nn


from hw8.roble.infrastructure import pytorch_util as ptu


class CQLCritic(BaseCritic):

    def __init__(self, hparams, optimizer_spec, **kwargs):
        super().__init__(**kwargs)
        self.env_name = hparams['env']['env_name']
        self.ob_dim = hparams['alg']['ob_dim']

        if isinstance(self.ob_dim, int):
            self.input_shape = (self.ob_dim,)
        else:
            self.input_shape = hparams['input_shape']

        self.ac_dim = hparams['alg']['ac_dim']
        self.double_q = hparams['alg']['double_q']
        self.grad_norm_clipping = hparams['alg']['grad_norm_clipping']
        self.gamma = hparams['alg']['gamma']

        self.optimizer_spec = optimizer_spec
        network_initializer = hparams['q_func']
        self.q_net = network_initializer(self.ob_dim, self.ac_dim)
        self.q_net_target = network_initializer(self.ob_dim, self.ac_dim)
        self.optimizer = self.optimizer_spec.constructor(
            self.q_net.parameters(),
            **self.optimizer_spec.optim_kwargs
        )
        self.learning_rate_scheduler = optim.lr_scheduler.LambdaLR(
            self.optimizer,
            self.optimizer_spec.learning_rate_schedule,
        )
        self.loss = nn.MSELoss()
        self.q_net.to(ptu.device)
        self.q_net_target.to(ptu.device)
        self.cql_alpha = hparams['alg'].get('cql_alpha', 0.0)

        print(f"DEBUG: CQLCritic Initialized - cql_alpha = {self.cql_alpha}")


    def dqn_loss(self, ob_no, ac_na, next_ob_no, reward_n, terminal_n):
        """ Implement DQN Loss """

        # Get current Q-values for chosen actions
        qa_t_values = self.q_net(ob_no).gather(1, ac_na.unsqueeze(1)).squeeze(1)
        
        # Compute target Q-values
        with torch.no_grad():
            if self.double_q:
                # Double Q-learning: use online network to select actions
                next_actions = self.q_net(next_ob_no).argmax(dim=1, keepdim=True)
                q_next = self.q_net_target(next_ob_no).gather(1, next_actions)
            else:
                # Regular DQN: use target network for action selection
                q_next = self.q_net_target(next_ob_no).max(dim=1)[0]
            
            target = reward_n + (1 - terminal_n) * self.gamma * q_next
        
        # Get Q-values for current states (for logging)
        q_t_values = self.q_net(ob_no)
        
        # Calculate MSE loss
        loss = self.loss(qa_t_values, target)
        
        return loss, qa_t_values, q_t_values


    def update(self, ob_no, ac_na, next_ob_no, reward_n, terminal_n):
        """
            Update the parameters of the critic.
            let sum_of_path_lengths be the sum of the lengths of the paths sampled from
                Agent.sample_trajectories
            let num_paths be the number of paths sampled from Agent.sample_trajectories
            arguments:
                ob_no: shape: (sum_of_path_lengths, ob_dim)
                next_ob_no: shape: (sum_of_path_lengths, ob_dim). The observation after taking one step forward
                reward_n: length: sum_of_path_lengths. Each element in reward_n is a scalar containing
                    the reward for each timestep
                terminal_n: length: sum_of_path_lengths. Each element in terminal_n is either 1 if the episode ended
                    at that timestep of 0 if the episode did not end
            returns:
                nothing
        """
        print("DEBUG: Entering CQL update function")

        ob_no = ptu.from_numpy(ob_no)
        ac_na = ptu.from_numpy(ac_na).to(torch.long)
        next_ob_no = ptu.from_numpy(next_ob_no)
        reward_n = ptu.from_numpy(reward_n)
        terminal_n = ptu.from_numpy(terminal_n)

        # Compute the DQN Loss 
        loss, qa_t_values, q_t_values = self.dqn_loss(
            ob_no, ac_na, next_ob_no, reward_n, terminal_n
            )
        
        # CQL Implementation
        # TODO: Implement CQL as described in the pdf and paper
        # Hint: After calculating cql_loss, augment the loss appropriately
        q_t = self.q_net(ob_no)
        q_t_logsumexp = torch.logsumexp(q_t, dim=1)
        current_action_q = q_t.gather(1, ac_na.unsqueeze(1)).squeeze(1)
        cql_loss = (q_t_logsumexp - current_action_q).mean() # CQL penalty term
        total_loss = loss + self.cql_alpha * cql_loss # Total loss = DQN loss + CQL penalty

        print(f"BEFORE UPDATE: Max Q-value: {q_t_values.max().item()}, Min Q-value: {q_t_values.min().item()}")

        self.optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), self.grad_norm_clipping)
        self.optimizer.step()

        print(f"AFTER UPDATE: Max Q-value: {q_t_values.max().item()}, Min Q-value: {q_t_values.min().item()}")

        info = {
        'Training Loss': ptu.to_numpy(total_loss),
        'CQL Loss': ptu.to_numpy(cql_loss),
        'Data q-values': ptu.to_numpy(current_action_q).mean(),
        'OOD q-values': ptu.to_numpy(q_t_logsumexp).mean(),
        'Max Q-value' : ptu.to_numpy(q_t_values.max())
        }       

        # TODO: Uncomment these lines after implementing CQL
        info['CQL Loss'] = ptu.to_numpy(cql_loss)
        info['Data q-values'] = ptu.to_numpy(current_action_q).mean()
        info['OOD q-values'] = ptu.to_numpy(q_t_logsumexp).mean()
        
        self.learning_rate_scheduler.step()
        print(f"CQL Loss: {cql_loss.item()}, Max Q-value: {q_t_values.max().item()}, Avg Q-value: {q_t_values.mean().item()}")


        return info

    def update_target_network(self):
        for target_param, param in zip(
                self.q_net_target.parameters(), self.q_net.parameters()
        ):
            target_param.data.copy_(param.data)

    def qa_values(self, obs):
        obs = ptu.from_numpy(obs)
        qa_values = self.q_net(obs)
        return ptu.to_numpy(qa_values)
