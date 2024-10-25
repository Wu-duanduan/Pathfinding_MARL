import numpy as np
from copy import deepcopy
from torch.optim import Adam
import torch
import core as core
from torch.utils.tensorboard import SummaryWriter

# from model import S, R
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ReplayBuffer:  # 采样后输出为tensor
    """
    A simple FIFO experience replay buffer for DDPG agents.
    """

    def __init__(self, obs_dim, act_dim, size):
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32, device=device) for k, v in batch.items()}


class DDPG:
    def __init__(self, obs_dim, act_dim, actor_critic=core.MLPActorCritic,
                 replay_size=int(1e5), gamma=0.99, polyak=0.995, pi_lr=1e-3, q_lr=1e-3, s_lr=1e-3, r_lr=1e-3,
                 act_noise=0.1):
        self.writer = SummaryWriter()

        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.gamma = gamma
        self.polyak = polyak
        self.act_noise = act_noise
        # torch.manual_seed(0)
        # np.random.seed(0)

        self.ac = actor_critic(obs_dim, act_dim).to(device)
        self.ac_targ = deepcopy(self.ac).to(device)

        self.pi_optimizer = Adam(self.ac.pi.parameters(), lr=pi_lr)
        self.q_optimizer = Adam(self.ac.q.parameters(), lr=q_lr)
        self.s_optimizer = Adam(self.ac.s.parameters(), lr=s_lr)
        self.r_optimizer = Adam(self.ac.r.parameters(), lr=r_lr)

        for p in self.ac_targ.parameters():
            p.requires_grad = False

        self.replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)

        self.update_num = 0

    def compute_loss_q(self, data, i_episodes):  # 返回(q网络loss, q网络输出的状态动作值即Q值)
        o0, a0, r0, o1, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']
        o1_values = self.ac.s(o0, a0)
        r0_values = self.ac.r(o0, a0)  # R值

        a1_values = self.ac.pi(o1_values)

        o2_values = self.ac.s(o1_values, a1_values)
        r1_values = self.ac.r(o1_values, a1_values)  # R+1值

        q0 = self.ac.q(o0, a0)
        q1 = self.ac.q(o1_values, a1_values)
        # Bellman backup for Q function
        with torch.no_grad():
            q_pi_targ0 = self.ac_targ.q(o1, self.ac_targ.pi(o1))
            backup0 = r0 + self.gamma * (1 - d) * q_pi_targ0
            q_pi_targ1 = self.ac_targ.q(o2_values, self.ac_targ.pi(o2_values))
            backup1 = r1_values + self.gamma * (1 - d) * q_pi_targ1
        # MSE loss against Bellman backup
        if i_episodes > -1:
            loss_q = ((q0 - backup0) ** 2 + (q1 - backup1) ** 2).mean()
        else:
            loss_q = ((q0 - backup0) ** 2).mean()
        loss_s = ((o1 - o1_values) ** 2).mean()
        loss_r = ((r0 - r0_values) ** 2).mean()

        return loss_q, loss_s, loss_r  # 这里的loss_q没加负号说明是最小化，很好理解，TD正是用函数逼近器去逼近backup，误差自然越小越好

    def compute_loss_pi(self, data):
        o = data['obs']
        q_pi = self.ac.q(o, self.ac.pi(o))
        return -q_pi.mean()  # 这里的负号表明是最大化q_pi,即最大化在当前state策略做出的action的Q值

    def update(self, data, i_episodes):
        # First run one gradient descent step for Q.
        loss_q, loss_s, loss_r = self.compute_loss_q(data, i_episodes)
        self.q_optimizer.zero_grad()
        loss_q.backward(retain_graph=True)
        self.q_optimizer.step()
        self.s_optimizer.zero_grad()
        loss_s.backward()
        self.s_optimizer.step()
        self.r_optimizer.zero_grad()
        loss_r.backward()
        self.r_optimizer.step()
        # Freeze Q-network so you don't waste computational effort
        # computing gradients for it during the policy learning step.
        for p in self.ac.q.parameters():
            p.requires_grad = False

        # Next run one gradient descent step for pi.
        self.pi_optimizer.zero_grad()
        loss_pi = self.compute_loss_pi(data)
        loss_pi.backward()
        self.pi_optimizer.step()

        self.update_num += 1
        self.writer.add_scalar('loss_Q', loss_q, self.update_num)
        self.writer.add_scalar('loss_pi', loss_pi, self.update_num)

        # Unfreeze Q-network so you can optimize it at next DDPG step.
        for p in self.ac.q.parameters():
            p.requires_grad = True

        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(self.ac.parameters(), self.ac_targ.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)
        return loss_s.detach().numpy(), loss_r.detach().numpy()

    def get_action(self, o, noise_scale):
        a = self.ac.act(torch.as_tensor(o, dtype=torch.float32, device=device))
        a += noise_scale * np.random.randn(self.act_dim)
        return np.clip(a, -1, 1)
