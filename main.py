#!/usr/bin/python

import matplotlib.pyplot as plt
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary
from arguments import parse_args
from IIFDS2 import IIFDS
from replay_buffer import ReplayBuffer
from model import MLPActor, MLPQFunction, MLPSFunction, MLPRFunction
from Method import getReward, setup_seed
from draw import Painter
import torch.nn.functional as F

def get_trainers(numberOfAgents, obs_shape_n, action_shape_n, arglist):
    """
    初始化每个agent的4个网络，用列表存储
    """
    actor_cur = [None for _ in range(numberOfAgents)]
    critics_cur = [None for _ in range(numberOfAgents)]
    actor_tar = [None for _ in range(numberOfAgents)]
    critics_tar = [None for _ in range(numberOfAgents)]
    optimizers_c = [None for _ in range(numberOfAgents)]
    optimizers_a = [None for _ in range(numberOfAgents)]

    for i in range(numberOfAgents):
        actor_cur[i] = MLPActor(obs_shape_n[i], action_shape_n[i],
                                [arglist.action_limit_min, arglist.action_limit_max]).to(arglist.device)
        critics_cur[i] = MLPQFunction(sum(obs_shape_n), sum(action_shape_n)).to(arglist.device)
        actor_tar[i] = MLPActor(obs_shape_n[i], action_shape_n[i],
                                [arglist.action_limit_min, arglist.action_limit_max]).to(arglist.device)
        critics_tar[i] = MLPQFunction(sum(obs_shape_n), sum(action_shape_n)).to(arglist.device)
        optimizers_a[i] = optim.Adam(actor_cur[i].parameters(), arglist.lr_a)
        optimizers_c[i] = optim.Adam(critics_cur[i].parameters(), arglist.lr_c)
    actor_tar = update_trainers(actor_cur, actor_tar, 1.0)  # just copy it
    critics_tar = update_trainers(critics_cur, critics_tar, 1.0)
    s_net = MLPSFunction(sum(obs_shape_n), sum(action_shape_n)).to(arglist.device)
    r_net = MLPRFunction(sum(obs_shape_n), sum(action_shape_n)).to(arglist.device)
    optimizer_s = optim.Adam(s_net.parameters(), 1e-3)
    optimizer_r = optim.Adam(r_net.parameters(), 1e-2)
    return actor_cur, critics_cur, actor_tar, critics_tar, optimizers_a, optimizers_c, s_net, r_net, optimizer_s, optimizer_r


def update_trainers(agents_cur, agents_tar, tao):
    """
    输入为：源网络和目标网络
    作用：使用soft update更新目标网络
    """
    for agent_c, agent_t in zip(agents_cur, agents_tar):
        key_list = list(agent_c.state_dict().keys())
        state_dict_t = agent_t.state_dict()
        state_dict_c = agent_c.state_dict()
        for key in key_list:
            state_dict_t[key] = state_dict_c[key] * tao + \
                                (1 - tao) * state_dict_t[key]
        agent_t.load_state_dict(state_dict_t)
    return agents_tar


def agents_train(arglist, episode_gone, game_step, update_cnt, memory, obs_size, action_size,
                 actors_cur, actors_tar, critics_cur, critics_tar, optimizers_a, optimizers_c, s_net, r_net,
                 optimizer_s, optimizer_r):
    """
    对每个agent的4个网络进行更新
    当game_step到达learning_start_step后开始更新
    每个agent都有一个critic，这个critic输入为所有agent的状态以及动作，而actor的输入只有agent自身的状态
    return: update_cnt(已经更新了的次数) actors_cur(actor网络) actors_target(actor目标网络) critics_cur(critic网络) critics_target(critic目标网络)
    """
    if game_step > arglist.learning_start_step and \
            (game_step - arglist.learning_start_step) % arglist.learning_fre == 0:
        if update_cnt == 0: print('\r=start training ...')
        update_cnt += 1

        for agent_idx, (actor_c, actor_t, critic_c, critic_t, opt_a, opt_c) in \
                enumerate(zip(actors_cur, actors_tar, critics_cur, critics_tar, optimizers_a, optimizers_c)):

            _obs_n_o, _action_n, _rew_n, _obs_n_n, _done_n = memory.sample(
                arglist.batch_size, agent_idx)  # 对每个agent在buffer中采样(buffer中存储的是所有agent的样本，可以使用agent_idx区分)
            rew = torch.tensor(_rew_n, device=arglist.device, dtype=torch.float)
            done_n = torch.tensor(_done_n, device=arglist.device, dtype=torch.float)
            action_cur_o = torch.from_numpy(_action_n).to(arglist.device, torch.float)
            obs_n_o = torch.from_numpy(_obs_n_o).to(arglist.device, torch.float)
            obs_n_n = torch.from_numpy(_obs_n_n).to(arglist.device, torch.float)
            obs_n_o1 = s_net(obs_n_o, action_cur_o)
            rew0 = r_net(obs_n_o, action_cur_o)

            action_cur_o1 = torch.cat([a_t(obs_n_o1[:, obs_size[idx][0]:obs_size[idx][1]]).detach() \
                                       for idx, a_t in enumerate(actors_cur)], dim=1)

            obs_n_o2 = s_net(obs_n_o1, action_cur_o1)
            rew1 = r_net(obs_n_o1, action_cur_o1)

            action_tar = torch.cat([a_t(obs_n_n[:, obs_size[idx][0]:obs_size[idx][1]]).detach() \
                                    for idx, a_t in enumerate(actors_tar)], dim=1)
            action_tar1 = torch.cat([a_t(obs_n_o2[:, obs_size[idx][0]:obs_size[idx][1]]).detach() \
                                     for idx, a_t in enumerate(actors_tar)], dim=1)
            q = critic_c(obs_n_o, action_cur_o).reshape(-1)  # q
            q1 = critic_c(obs_n_o1, action_cur_o1).reshape(-1)  # q
            # 这里放在no_grad下与否并不会影响更新，但是会节省速度，否则loss会传到actor和target_critic中并更新它们的梯度，这是没必要的！
            with torch.no_grad():
                q_ = critic_t(obs_n_n, action_tar).reshape(-1)  # q_
                tar_value = q_ * arglist.gamma * (1 - done_n) + rew  # q_*gamma*done + reward
                q_1 = critic_t(obs_n_o2, action_tar1).reshape(-1)  # q_
                tar_value1 = q_1 * arglist.gamma * (1 - done_n) + rew1  # q_*gamma*done + reward
            if episode_gone > -1:
                loss_c = torch.mean(F.mse_loss(q, tar_value) + F.mse_loss(q1, tar_value1))  # bellman equation
            else:
                loss_c = torch.mean(F.mse_loss(q, tar_value))  # bellman equation
            opt_c.zero_grad()
            loss_c.backward(retain_graph=True)
            # nn.utils.clip_grad_norm_(critic_c.parameters(), arglist.max_grad_norm)  # 限制梯度最大值
            opt_c.step()
            loss_s = torch.mean(F.mse_loss(obs_n_n, obs_n_o1))
            optimizer_s.zero_grad()
            loss_s.backward()
            # nn.utils.clip_grad_norm_(critic_c.parameters(), arglist.max_grad_norm)  # 限制梯度最大值
            optimizer_s.step()
            loss_r = torch.mean(F.mse_loss(rew, rew0))
            optimizer_r.zero_grad()
            loss_r.backward()
            # nn.utils.clip_grad_norm_(critic_c.parameters(), arglist.max_grad_norm)  # 限制梯度最大值
            optimizer_r.step()
            # --use the data to update the ACTOR
            # There is no need to cal other agent's action
            policy_c_new = actor_c(obs_n_o[:, obs_size[agent_idx][0]:obs_size[agent_idx][1]])
            # update the aciton of this agent
            action_cur_o[:, action_size[agent_idx][0]:action_size[agent_idx][1]] = policy_c_new
            loss_a = torch.mul(-1, torch.mean(critic_c(obs_n_o, action_cur_o)))  # 乘以负一最小化就是最大化critic的输出值，这是actor的更新方向

            opt_a.zero_grad()
            loss_a.backward()
            # nn.utils.clip_grad_norm_(actor_c.parameters(), arglist.max_grad_norm)
            opt_a.step()

        # 使用soft update对target网络部分更新
        actors_tar = update_trainers(actors_cur, actors_tar, arglist.tao)
        critics_tar = update_trainers(critics_cur, critics_tar, arglist.tao)
    else:
        for agent_idx, (actor_c, actor_t, critic_c, critic_t, opt_a, opt_c) in \
                enumerate(zip(actors_cur, actors_tar, critics_cur, critics_tar, optimizers_a, optimizers_c)):
            _obs_n_o, _action_n, _rew_n, _obs_n_n, _done_n = memory.sample(
                arglist.batch_size, agent_idx)  # 对每个agent在buffer中采样(buffer中存储的是所有agent的样本，可以使用agent_idx区分)
            rew = torch.tensor(_rew_n, device=arglist.device, dtype=torch.float)
            done_n = torch.tensor(_done_n, device=arglist.device, dtype=torch.float)
            action_cur_o = torch.from_numpy(_action_n).to(arglist.device, torch.float)
            obs_n_o = torch.from_numpy(_obs_n_o).to(arglist.device, torch.float)
            obs_n_n = torch.from_numpy(_obs_n_n).to(arglist.device, torch.float)

            obs_n_o1 = s_net(obs_n_o, action_cur_o)
            rew0 = r_net(obs_n_o, action_cur_o)

            loss_s = torch.mean(F.mse_loss(obs_n_n, obs_n_o1))
            optimizer_s.zero_grad()
            loss_s.backward()
            # nn.utils.clip_grad_norm_(critic_c.parameters(), arglist.max_grad_norm)  # 限制梯度最大值
            optimizer_s.step()
            loss_r = torch.mean(F.mse_loss(rew, rew0))
            optimizer_r.zero_grad()
            loss_r.backward()
            # nn.utils.clip_grad_norm_(critic_c.parameters(), arglist.max_grad_norm)  # 限制梯度最大值
            optimizer_r.step()
    return update_cnt, actors_cur, actors_tar, critics_cur, critics_tar, s_net, r_net, loss_s.detach().numpy(), loss_r.detach().numpy()


def train(arglist):
    """
    arglist: 参数表
    这是核心训练函数
    """
    """step1: create the enviroment"""
    iifds = IIFDS()
    """step2: create angents"""
    obs_shape_n = [21 for i in range(iifds.numberofuav)]  # 每个agent states维度为3 为航迹上一点到某一个障碍物中心的向量坐标
    action_shape_n = [3 for i in range(iifds.numberofuav)]  # 每个无人机的航向角
    actors_cur, critics_cur, actors_tar, critics_tar, optimizers_a, optimizers_c, s_net, r_net, optimizer_s, optimizer_r = \
        get_trainers(iifds.numberofuav, obs_shape_n, action_shape_n, arglist)
    memory = ReplayBuffer(arglist.memory_size)
    """step3: init the pars"""
    obs_size = []  # 这个是因为存储样本的是一个buffer，需要对每个agent的区域进行划分
    action_size = []  # 这个是因为存储样本的是一个buffer，需要对每个agent的区域进行划分
    head_o, head_a, end_o, end_a = 0, 0, 0, 0
    for obs_shape, action_shape in zip(obs_shape_n, action_shape_n):  # 用一个buffer储存所有信息，下面是计算各个agent的状态和动作的索引
        end_o = end_o + obs_shape
        end_a = end_a + action_shape
        range_o = (head_o, end_o)
        range_a = (head_a, end_a)
        obs_size.append(range_o)
        action_size.append(range_a)
        head_o = end_o
        head_a = end_a

    print('=3 starting iterations ...')
    rewardList = []
    maxReward = -np.inf
    var = 2
    game_step = 0
    update_cnt = 0
    s_loss_list = []
    r_loss_list = []
    for episode_gone in range(arglist.max_episode):
        start = iifds.start
        q = []
        qBefore = []
        goal = iifds.goal
        for i in range(8):
            q.append(iifds.start[i] + np.random.random(3) * 3)
            qBefore.append([None, None, None])
        iifds.reset()
        rewardSum = 0
        pathLength = [0, 0, 0, 0, 0, 0, 0, 0]
        s_loss = 0
        r_loss = 0
        for episode_cnt in range(arglist.per_episode_max_len):
            dic1, dic2 = iifds.updateObs()
            vObs1, obsCenter1, obsCenterNext1 = dic1['v'], dic1['obsCenter'], dic1['obsCenterNext']
            vObs2, obsCenter2, obsCenterNext2 = dic2['v'], dic2['obsCenter'], dic2['obsCenterNext']
            obsCenter = [obsCenter1, obsCenter2]
            vObs = [vObs1, vObs2]
            obsCenterNext = [obsCenterNext1, obsCenterNext2]

            obsDicq = iifds.calDynamicState(q, obsCenter)  # 相对位置字典
            obs_n_uav1, obs_n_uav2, obs_n_uav3, obs_n_uav4, obs_n_uav5, obs_n_uav6, obs_n_uav7, obs_n_uav8 = \
                obsDicq['uav1'], obsDicq['uav2'], obsDicq['uav3'], obsDicq['uav4'], obsDicq['uav5'], obsDicq['uav6'], \
                obsDicq['uav7'], obsDicq['uav8']
            obs_n = obs_n_uav1 + obs_n_uav2 + obs_n_uav3 + obs_n_uav4 + obs_n_uav5 + obs_n_uav6 + obs_n_uav7 + obs_n_uav8
            # get action
            if episode_gone > arglist.actor_begin_work:
                if episode_gone == arglist.actor_begin_work + 1 and episode_cnt == 0: print('==actor begin to work.')
                if var <= 0.10:
                    var = 0.10  # 保证后期的一定探索性
                else:
                    var *= 0.9999
                action_n = [agent(torch.from_numpy(obs).to(arglist.device, torch.float)).detach().cpu().numpy() \
                            for agent, obs in zip(actors_cur, obs_n)]
                # 前iifds.numberOfSphere个是sphere的action，后面是cylinder的action，最后是cone的action
                action_n = np.clip(np.random.normal(action_n, var), arglist.action_limit_min, arglist.action_limit_max)
                action_n = action_n.reshape(-1)
            else:  # 随机采样
                if episode_gone == 0 and episode_cnt == 0: print(
                    '==this period of time the actions will be sampled randomly')
                action_n = [random.uniform(arglist.action_limit_min, arglist.action_limit_max) \
                            for _ in range(3 * iifds.numberofuav)]
            # interact with enviroment
            qNext = []
            for i in range(8):
                qNext.append(iifds.getqNext(i, q, obsCenter, vObs, action_n[3 * i], action_n[3 * i + 1],
                                            action_n[3 * i + 2], qBefore, goal))

            # flag = iifds.checkCollision(qNext)
            obsDicqNext = iifds.calDynamicState(qNext, obsCenterNext)
            new_obs_n_uav1, new_obs_n_uav2, new_obs_n_uav3, new_obs_n_uav4, new_obs_n_uav5, new_obs_n_uav6, new_obs_n_uav7, new_obs_n_uav8 = \
                obsDicqNext['uav1'], obsDicqNext['uav2'], obsDicqNext['uav3'], obsDicqNext['uav4'], obsDicqNext['uav5'], \
                obsDicqNext['uav6'], \
                obsDicqNext['uav7'], obsDicqNext['uav8']
            new_obs_n = new_obs_n_uav1 + new_obs_n_uav2 + new_obs_n_uav3 + new_obs_n_uav4 + new_obs_n_uav5 + new_obs_n_uav6 + new_obs_n_uav7 + new_obs_n_uav8

            done_n = [
                True if ((iifds.distanceCost(goal[0], qNext[0]) < iifds.threshold) and (iifds.distanceCost(goal[1],
                                                                                                           qNext[
                                                                                                               1]) < iifds.threshold) and (
                                     iifds.distanceCost(goal[2], qNext[2]) < iifds.threshold) and \
                         (iifds.distanceCost(goal[3], qNext[3]) < iifds.threshold) and (
                                     iifds.distanceCost(goal[4], qNext[4]) < iifds.threshold) \
                         and (iifds.distanceCost(goal[5], qNext[5]) < iifds.threshold) and (
                                     iifds.distanceCost(goal[6], qNext[6]) < iifds.threshold) \
                         and (iifds.distanceCost(goal[7], qNext[7]) < iifds.threshold)) else False \
                for _ in range(iifds.numberofuav)]
            for i in range(iifds.numberofuav):
                pathLength[i] += iifds.distanceCost(q[i], qNext[i])

            rew_n = [getReward(obsCenterNext, qNext, q, qBefore, goal, iifds) for _ in
                     range(iifds.numberofuav)]  # 每个agent使用相同的reward
            rewardSum += rew_n[0]
            # save the experience
            memory.add(obs_n, action_n, rew_n, new_obs_n, done_n)

            # train our agents
            update_cnt, actors_cur, actors_tar, critics_cur, critics_tar, s_net, r_net, loss_s, loss_r = agents_train(
                arglist, episode_gone, game_step, update_cnt, memory, obs_size, action_size,
                actors_cur, actors_tar, critics_cur, critics_tar, optimizers_a, optimizers_c, s_net, r_net, optimizer_s,
                optimizer_r)
            s_loss += loss_s
            r_loss += loss_r
            # update the position
            game_step += 1
            qBefore = q
            q = qNext
            if all(done_n):  # done_n全是True all的结果为True
                for i in range(iifds.numberofuav):
                    pathLength[i] += iifds.distanceCost(q[i], goal[i])
                break
        s_loss_list.append(s_loss)
        r_loss_list.append(r_loss)
        print('Episode:', episode_gone, 'Reward:%f' % rewardSum, 'var:%f' % var, 'update_cnt:%d' % update_cnt)
        rewardList.append(round(rewardSum, 2))
        if episode_gone > arglist.max_episode * 2 / 3:
            if rewardSum > maxReward:
                maxReward = rewardSum
                print("历史最优reward，已保存模型！")
                for idx, pi in enumerate(actors_cur):
                    torch.save(pi, 'TrainedModel/Actor.%d.pkl' % idx)

    episodes_list = np.arange(np.array(rewardList).shape[0])
    np.save('MADDPG_episodes.npy', episodes_list)
    np.save('MADDPG_mv_return11.npy', rewardList)
    np.save('MADDPG_s_loss11.npy', s_loss_list)
    np.save('MADDPG_r_loss11.npy', r_loss_list)


if __name__ == '__main__':
    setup_seed(5)
    arglist = parse_args()
    train(arglist)
