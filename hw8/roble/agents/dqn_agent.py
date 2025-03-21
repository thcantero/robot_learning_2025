import numpy as np

import torch
from hw8.roble.infrastructure.dqn_utils import OptimizerSpec

from hw8.roble.infrastructure.dqn_utils import MemoryOptimizedReplayBuffer, PiecewiseSchedule
from hw8.roble.policies.argmax_policy import ArgMaxPolicy
from hw8.roble.critics.dqn_critic import DQNCritic

def parse_optimizer_spec(spec_dict):
    """
    Convert a dictionary with:
      {
        "constructor": "Adam" or "SGD",
        "optim_kwargs": {...},
        "learning_rate_schedule": "lambda t: 1.0"
      }
    into an actual OptimizerSpec with a Python function for constructor.
    """
    if spec_dict["constructor"] == "Adam":
        constructor_fn = torch.optim.Adam
    elif spec_dict["constructor"] == "SGD":
        constructor_fn = torch.optim.SGD
    else:
        raise ValueError(f"Unknown constructor: {spec_dict['constructor']}")

    schedule_str = spec_dict["learning_rate_schedule"]
    schedule_fn = eval(schedule_str)  # e.g. "lambda t: 1.0" -> lambda t: 1.0

    return OptimizerSpec(
        constructor=constructor_fn,
        optim_kwargs=spec_dict["optim_kwargs"],
        learning_rate_schedule=schedule_fn
    )

class DQNAgent(object):
    def __init__(self, env, agent_params):

        # Debug: Check the type of agent_params['alg'] before using it
        print("DEBUG: agent_params['alg'] type inside DQNAgent (before fix) =", type(agent_params['alg']))
        print("DEBUG: agent_params['alg'] contents inside DQNAgent (before fix) =", agent_params['alg'])

        # Fix: If agent_params['alg'] is a tuple, extract the dictionary
        if isinstance(agent_params['alg'], tuple):
            print("WARNING: Detected 'alg' as a tuple inside DQNAgent. Fixing...")
            agent_params['alg'] = agent_params['alg'][0]  # Convert it back to a dictionary

        # Debug: Verify the fix
        print("DEBUG: agent_params['alg'] type inside DQNAgent (after fix) =", type(agent_params['alg']))
        print("DEBUG: agent_params['alg'] contents inside DQNAgent (after fix) =", agent_params['alg'])

        self.env = env
        self.agent_params = agent_params
        
        self.batch_size = agent_params['alg']['batch_size']
        self.last_obs = self.env.reset()
        self.num_actions = agent_params['alg']['ac_dim']
        self.learning_starts = agent_params['alg']['learning_starts']
        self.learning_freq = agent_params['alg']['learning_freq']
        self.target_update_freq = agent_params['alg']['target_update_freq']

        self.replay_buffer_idx = None
        self.exploration_initial_eps = agent_params['alg'].get('exploration_initial_eps', 1.0)
        self.exploration_final_eps = agent_params['alg'].get('exploration_final_eps', 0.01)
        self.exploration_decay_steps = agent_params['alg'].get('exploration_decay_steps', 1e6)

        from hw8.roble.infrastructure.dqn_utils import LinearSchedule
        self.exploration_schedule = LinearSchedule(
            schedule_timesteps=self.exploration_decay_steps,
            initial_p=self.exploration_initial_eps,
            final_p=self.exploration_final_eps
        )
        
        self.optimizer_spec = agent_params['alg']['optimizer_spec']

        # Build the critic
        alg_dict = agent_params["alg"]

        critic_params = {
            "ob_dim": alg_dict["ob_dim"],
            "ac_dim": alg_dict["ac_dim"],
            "double_q": alg_dict["double_q"],
            "grad_norm_clipping": alg_dict["grad_norm_clipping"],
            "gamma": alg_dict["gamma"],
            "optimizer_spec": self.optimizer_spec,
            #"optimizer_spec": alg_dict["optimizer_spec"],        
            "q_func": agent_params["q_func"],                       
            "input_shape": agent_params.get("input_shape", None),             
        }

        self.critic = DQNCritic(**critic_params)
        self.actor = ArgMaxPolicy(self.critic)

        lander = agent_params['env']['env_name'].startswith('LunarLander')
        self.replay_buffer = MemoryOptimizedReplayBuffer(
            agent_params['replay_buffer_size'], agent_params['frame_history_len'], lander=lander)
        
        self.t = 0
        self.num_param_updates = 0

    def add_to_replay_buffer(self, paths):
        for path in paths:
            observations = path["observation"]       # shape [path_len+1, ob_dim]
            actions = path["action"]                 # shape [path_len]
            rewards = path["reward"]                 # shape [path_len]
            next_observations = path["next_observation"]  # shape [path_len, ob_dim]
            terminals = path["terminal"]             # shape [path_len], 0 or 1

        # Iterate through each step in the path
        for t in range(len(actions)):
            # store_frame returns an idx
            self.replay_buffer_idx = self.replay_buffer.store_frame(observations[t])

            # Store the effect of your action: (action, reward, done)
            self.replay_buffer.store_effect(
                self.replay_buffer_idx,
                actions[t],
                rewards[t],
                terminals[t]
            )

    def step_env(self):
        """
            Step the env and store the transition
            At the end of this block of code, the simulator should have been
            advanced one step, and the replay buffer should contain one more transition.
            Note that self.last_obs must always point to the new latest observation.
        """
        #print(f"[DEBUG] step_env() last_obs shape: {self.last_obs.shape}, dtype: {self.last_obs.dtype}")


        #print("[dqn_agent.step_env] self.last_obs shape:", self.last_obs.shape)
        # 1) store current single frame
        idx = self.replay_buffer.store_frame(self.last_obs)

        # 2) build 4-frame stacked obs from replay buffer
        stacked_obs = self.replay_buffer.encode_recent_observation()

        # 3) choose action
        if self.t < self.learning_starts:
            action = self.env.action_space.sample()
        else:
            # Use ε-greedy exploration
            epsilon = self.exploration_schedule.value(self.t)
            if np.random.rand() < epsilon:
                action = self.env.action_space.sample()
            else:
                action = self.actor.get_action(self.last_obs)

        # 4) step environment
        next_obs, reward, done, info = self.env.step(action)

        # 5) store effect
        self.replay_buffer.store_effect(idx, action, reward, done)

        # 6) if done, reset
        if done:
            self.last_obs = self.env.reset()
        else:
            self.last_obs = next_obs

        self.t += 1

    ####################################
    ####################################

    def sample(self, batch_size):
        if self.replay_buffer.can_sample(batch_size):
            return self.replay_buffer.sample(batch_size)
        else:
            return [],[],[],[],[]

    def train(self, ob_no, ac_na, re_n, next_ob_no, terminal_n):
        # if there are not enough samples
        if self.t < self.learning_starts or len(ob_no)==0:
            return {}

        ob_no       = np.array(ob_no)       
        ac_na       = np.array(ac_na)
        re_n        = np.array(re_n)
        next_ob_no  = np.array(next_ob_no)
        terminal_n  = np.array(terminal_n)

        # Update the Q-network
        loss_info = self.critic.update(ob_no, ac_na, next_ob_no, re_n, terminal_n)

        # Periodically update target network
        if self.num_param_updates % self.target_update_freq == 0:
            self.critic.update_target_network()

        self.num_param_updates += 1

        return loss_info

    ####################################
    ####################################