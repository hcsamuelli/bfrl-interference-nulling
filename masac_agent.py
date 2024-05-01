# %%
import numpy as np
import torch
from torch import log_, nn, optim
import torch.nn.functional as F

from pettingzoo import AECEnv
from stable_baselines3.common.buffers import ReplayBuffer


# %%
class FCActor(nn.Module):
    def __init__(self, state_dim, action_dim, min_action, max_action):
        super().__init__()
        self.LOG_STD_MAX = 2
        self.LOG_STD_MIN = -5
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc_mean = nn.Linear(256, action_dim)
        self.fc_logstd = nn.Linear(256, action_dim)
        self.register_buffer(
            "action_scale",
            torch.tensor(
                (max_action - min_action) / 2.0,
                dtype=torch.float32,
            ).view(-1),
        )
        self.register_buffer(
            "action_bias",
            torch.tensor(
                (max_action + min_action) / 2.0,
                dtype=torch.float32,
            ).view(-1),
        )
        print("Initialized FCActor, input shape:", state_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = self.LOG_STD_MIN + 0.5 * (self.LOG_STD_MAX - self.LOG_STD_MIN) * (
            log_std + 1
        )  # From SpinUp / Denis Yarats
        return mean, log_std


    def get_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)

        # for reparameterization trick (mean + std * N(0,1))
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)

        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean


class SoftQNet(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim + action_dim, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 1)
        print(
            f"Initialized SoftQNet, input shape: {obs_dim} + {action_dim} = {obs_dim + action_dim}"
        )

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


class Agent:
    def __init__(self, env: AECEnv, agent, args, Actor=FCActor, Critic=SoftQNet):
        print(f"Initializing Agent {agent}...")
        a = env.agents[agent]
        obs_dim = np.prod(env.observation_spaces[a].shape)
        action_dim = np.prod(env.action_spaces[a].shape)
        min_action = env.action_spaces[a].low
        max_action = env.action_spaces[a].high

        self.actor = Actor(obs_dim, action_dim, min_action, max_action).to(args.device)
        self.qf1 = Critic(obs_dim, action_dim).to(args.device)
        self.qf2 = Critic(obs_dim, action_dim).to(args.device)

        self.qf1_target = Critic(obs_dim, action_dim).to(args.device)
        self.qf2_target = Critic(obs_dim, action_dim).to(args.device)
        self.qf1_target.load_state_dict(self.qf1.state_dict())
        self.qf2_target.load_state_dict(self.qf2.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=args.policy_lr)
        self.qf_optimizer = optim.Adam(
            list(self.qf1.parameters()) + list(self.qf2.parameters()), lr=args.q_lr
        )

        self.rb = ReplayBuffer(
            args.buffer_size,
            env.observation_spaces[a],
            env.action_spaces[a],
            args.device,
            handle_timeout_termination=False,
        )

        if args.autotune:
            self.target_entropy = (
                -torch.prod(torch.Tensor(env.action_spaces[a].shape))
                .to(args.device)
                .item()
            )
            self.log_alpha = torch.zeros(1, requires_grad=True, device=args.device)
            self.alpha = self.log_alpha.exp()
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=args.q_lr)
        else:
            self.alpha = args.alpha

        print("=====================================")
