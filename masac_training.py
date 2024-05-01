# %%
import os
import sys
import random
import time
from datetime import datetime
from dataclasses import dataclass
import pickle as pkl
from tracemalloc import start

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard.writer import SummaryWriter

import gymnasium as gym
import pettingzoo
from stable_baselines3.common.buffers import ReplayBuffer

from mimopy import AntennaArray, Network
from mimopy.channels import LoS

from maenv import MIMOMAEnv
from masac_agent import FCActor, SoftQNet, Agent
from env_setup import *

IPYTHON = "IPython" in sys.modules
if IPYTHON:
    from IPython.core.getipython import get_ipython

    get_ipython().run_line_magic("reload_ext", "autoreload")
    get_ipython().run_line_magic("autoreload", "2")
    get_ipython().run_line_magic("config", "InlineBackend.figure_format = 'retina'")

@dataclass
class SAC_Args:
    exp_name: str = "masac"
    env_id: str = "DL"
    # metrics: str = "spectral_efficiency"
    # target_meas: float = 2.0
    metrics: str = "sinr"
    target_meas: float = 18.0

    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    cuda_id: int = 1

    # Algorithm specific arguments
    total_timesteps: int = 100000
    buffer_size: int = int(1e6)
    gamma: float = 0.7
    """the discount factor gamma, default is 0.99"""
    tau: float = 0.005
    """target smoothing coefficient (default: 0.005)"""
    batch_size: int = 2048
    learning_starts: int = int(5e3)
    """timestep to start learning"""
    policy_lr: float = 3e-4
    """the learning rate of the policy network optimizer"""
    q_lr: float = 1e-3
    """the learning rate of the Q network network optimizer"""
    policy_frequency: int = 2
    """the frequency of training policy (delayed)"""
    target_network_frequency: int = 1  # Denis Yarats' implementation delays this by 2.
    """the frequency of updates for the target nerworks"""
    noise_clip: float = 0.5
    """noise clip parameter of the Target Policy Smoothing Regularization"""
    alpha: float = 0.1
    """Entropy regularization coefficient."""
    # autotune: bool = True
    autotune: bool = False
    """automatic tuning of the entropy coefficient"""
    save_model: bool = True
    """whether to save the model"""
    load_model: any = None
    """the path to load the model from"""

    # env specific arguments
    Nt: int = 8
    Nr: int = 8
    Ndl: int = 1
    Nul: int = 1

    tx_coord: tuple = (0, 1000, 0)
    rx_coord: tuple = (0, -1000, 0)
    dl_coord: tuple = (5000, 6000, 0)
    ul_coord: tuple = (5000, -6000, 0)

    dl_noise: float = 0
    rx_noise: float = 0

    tx_power: float = 8.0
    rx_power: float = 8.0
    dl_power: float = 8.0
    ul_power: float = 8.0

    tolerance: float = 1
    tolerance_decay: float = 0.1
    # metrics: str = "sinr"
    # target_meas: float = 19.0


if __name__ == "__main__":
    args = SAC_Args()

    # Cuda device
    args.device = torch.device(
        "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    )
    print(f"Device: {args.device}")

    # SEEDING
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    # Environment
    # env = DL_network() if args.env_id == "DL" else FD_network()
    env = DL_network(args)
    net = env.network

    # Directory and device setup
    print(f"Target measurements: {args.target_meas:.3f}")
    agents = {a: Agent(env, a, args) for a in env.agents}
    print(f"Initialized agents {env.agents}")

    # %%
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    run_name = (
        f"{args.env_id}/{args.exp_name}/{timestamp}_{args.seed}_{args.dl_noise}db"
    )
    # root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    root_dir = os.path.dirname(__file__)
    run_dir = f"{root_dir}/runs/{run_name}"
    if not os.path.exists(run_dir):
        os.makedirs(run_dir)
        for a in env.agents:
            os.makedirs(f"{run_dir}/agent_{a}", exist_ok=True)
    print(f"Save directory: {run_dir}")

    # Tensorboard writer
    writer = SummaryWriter(run_dir)
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    start_time = time.time()
    obs = env.observe(env.agent_selection)

    for global_step, a in enumerate(env.agent_iter(args.total_timesteps)):
        agent = agents[a]
        # ALGO Logic
        if global_step < args.learning_starts:
            action = env.action_spaces[a].sample()
        else:
            action, _, _ = agent.actor.get_action(torch.Tensor(obs).to(args.device))
            action = action.detach().cpu().numpy()

        # Environment step
        env.step(action)
        next_obs, rewards, terminations, truncations, infos = env.last()
        real_next_obs = next_obs.copy()
        agent.rb.add(obs, next_obs, action, rewards, terminations, infos)

        # TODO: handle 'final observations' if needed

        obs = next_obs

        # Train the agent
        if global_step >= args.learning_starts:
            data = agent.rb.sample(args.batch_size)
            with torch.no_grad():
                next_state_action, next_state_log_pi, _ = agent.actor.get_action(
                    data.next_observations
                )
                qf1_next_target = agent.qf1_target(
                    data.next_observations, next_state_action
                )
                qf2_next_target = agent.qf2_target(
                    data.next_observations, next_state_action
                )
                min_qf_next_target = (
                    torch.min(qf1_next_target, qf2_next_target)
                    - agent.alpha * next_state_log_pi
                )
                next_q_value = data.rewards.flatten() + (
                    1 - data.dones.flatten()
                ) * args.gamma * (min_qf_next_target).view(-1)

            qf1_a_values = agent.qf1(data.observations, data.actions).view(-1)
            qf2_a_values = agent.qf2(data.observations, data.actions).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
            qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
            qf_loss = qf1_loss + qf2_loss

            agent.qf_optimizer.zero_grad()
            qf_loss.backward()
            agent.qf_optimizer.step()

            # if (global_step - args.learning_starts - 1999) % 2000 == 0:
            #     env.reduce_action_space(0.7)
            #     print(
            #         f"Max amp/phase change reduced to {env.max_amp_change}, {env.max_phase_change * 180 / np.pi:.2f} deg"
            #     )

            # TD3 delayed policy updates
            if global_step % args.policy_frequency == 0:
                # compensate for the delay by doing 'actor_update_interval' instead of 1
                for _ in range(args.policy_frequency):
                    pi, log_pi, _ = agent.actor.get_action(data.observations)
                    qf1_pi = agent.qf1(data.observations, pi)
                    qf2_pi = agent.qf2(data.observations, pi)
                    min_qf_pi = torch.min(qf1_pi, qf2_pi)
                    actor_loss = (agent.alpha * log_pi - min_qf_pi).mean()

                    agent.actor_optimizer.zero_grad()
                    actor_loss.backward()
                    agent.actor_optimizer.step()

                    if args.autotune:
                        with torch.no_grad():
                            _, log_pi, _ = agent.actor.get_action(data.observations)
                        alpha_loss = (
                            -agent.log_alpha.exp() * (log_pi + agent.target_entropy)
                        ).mean()

                        agent.alpha_optimizer.zero_grad()
                        alpha_loss.backward()
                        agent.alpha_optimizer.step()
                        agent.alpha = agent.log_alpha.exp().item()

            # Update target networks
            if global_step % args.target_network_frequency == 0:
                for param, target_param in zip(
                    agent.qf1.parameters(), agent.qf1_target.parameters()
                ):
                    target_param.data.copy_(
                        args.tau * param.data + (1 - args.tau) * target_param.data
                    )
                for param, target_param in zip(
                    agent.qf2.parameters(), agent.qf2_target.parameters()
                ):
                    target_param.data.copy_(
                        args.tau * param.data + (1 - args.tau) * target_param.data
                    )

            # Log losses to tensorboard
            if global_step % 51 == 0:
                # global_step = float(global_step)

                for a in env.agents:
                    # Network related metrics
                    for key in ("sinr", "spectral_efficiency", "gain"):
                        writer.add_scalar(
                            f"A{a} Current State/{key}",
                            env.state[key][a],
                            global_step,
                        )
                        writer.add_scalar(
                            f"A{a} Best State/{key}",
                            env.best_state[key][a],
                            global_step,
                        )
                # Agent related metrics
                a = env.agent_selection
                writer.add_scalar(
                    f"A{a} Losses/alpha",
                    agents[a].alpha,
                    global_step,
                )
                if args.autotune:
                    writer.add_scalar(
                        f"A{a} Losses/alpha loss",
                        alpha_loss.item(),
                        global_step,
                    )
                writer.add_scalar(
                    f"A{a} Losses/qf1_values",
                    qf1_a_values.mean().item(),
                    global_step,
                )
                writer.add_scalar(
                    f"A{a} Losses/qf2_values",
                    qf2_a_values.mean().item(),
                    global_step,
                )
                writer.add_scalar(
                    f"A{a} Losses/qf_loss", qf_loss.item() / 2.0, global_step
                )
                writer.add_scalar(
                    f"A{a} Losses/actor_loss", actor_loss.item(), global_step
                )
                writer.add_scalar(f"A{a} Losses/rewards", rewards, global_step)
                writer.add_scalar(
                    "SPS", global_step / (time.time() - start_time), global_step
                )

            if global_step % 500 == 0:
                with open(f"{run_dir}/env.pkl", "wb") as f:
                    pkl.dump(env, f)
                if args.save_model:
                    for a in env.agents:
                        agent = agents[a]
                        torch.save(
                            agent.actor.state_dict(),
                            f"{run_dir}/agent_{a}/actor_{global_step}.pth",
                        )
                        torch.save(
                            agent.qf1.state_dict(),
                            f"{run_dir}/agent_{a}/qf1_{global_step}.pth",
                        )
                        torch.save(
                            agent.qf2.state_dict(),
                            f"{run_dir}/agent_{a}/qf2_{global_step}.pth",
                        )

                print("==========================")
                print("Step:", global_step)
                print("SPS:", int(global_step / (time.time() - start_time)))
                print("Agent: ", env.agent_selection)
                print("Rewards:", rewards)
                print(
                    f"Max amp/phase change: {env.max_amp_change}, {env.max_phase_change * 180 / np.pi:.2f} deg"
                )
                print(f"target {args.metrics}, {env.target_meas}")
                print(
                    "Target dist:",
                    np.asarray(env.target_meas) - np.asarray(env.state[env.metrics]),
                )
                print("-------Best State-------")
                for key in ["sinr", "spectral_efficiency", "gain"]:
                    print(f"{key}:", env.best_state[key])
                print("-------Current State---------")
                for key in ["sinr", "spectral_efficiency", "gain"]:
                    print(f"{key}:", env.state[key])

                # print("gain:", infos["gain"][0])
                if IPYTHON:
                    plt.close("all")
                    # env.net.plot_gain(weights=env.best_weights, figsize=(10, 3))
                    # env.net.plot_gain(figsize=(10, 3))
                    fig, axes = plt.subplots(1, 2, figsize=(10, 5), dpi=300, subplot_kw={"projection": "polar"})
                    net.nodes["tx"].plot_gain(True, weights=env.best_weights, ax=axes[0])
                    net.nodes["tx"].plot_gain(True, ax=axes[1])
                    axes[0].set_title("Best Gain")
                    axes[1].set_title("Current Gain")
                    plt.tight_layout()
                    plt.show()
# %%
