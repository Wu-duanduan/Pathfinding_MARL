#!/usr/bin/python

import matplotlib.pyplot as plt
import numpy as np
import torch
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def getReward(obsCenterNext, qNext, q, qBefore, goal, iifds):
    """
    获取强化学习奖励值函数
    """
    distance = []
    obsR = []
    distances_g = []
    threshold = []
    flag_ = 0
    for i in range(8):
        distances_g.append(iifds.distanceCost(iifds.start[i], goal[i]))
        threshold.append(iifds.threshold)
        for j in range(2):
            distance.append(iifds.distanceCost(qNext[i][0:2], obsCenterNext[j][0:2]))
            obsR.append(iifds.uavR + iifds.cylinderR)
            if iifds.distanceCost(qNext[i][0:2], obsCenterNext[j][0:2]) <= iifds.uavR + iifds.cylinderR:
                flag_ = 1
    for i in range(8):
        j = i + 1
        while j < 8:
            distance.append(iifds.distanceCost(qNext[i], qNext[j]))
            obsR.append(2 * iifds.uavR)
            if iifds.distanceCost(qNext[i], qNext[j]) <= 2 * iifds.uavR:
                flag_ = 1
            j += 1

    flag = True if flag_ == 1 else False
    rewardsum = 0
    rewarduav = []
    dis_len = len(distance)
    if flag:  # 与障碍物有交点
        for i in range(dis_len):
            if i < 16 and distance[i] <= iifds.uavR + iifds.cylinderR:
                rewardsum += (distance[i] - (iifds.uavR + iifds.cylinderR)) / (iifds.uavR + iifds.cylinderR) - 1
            elif i >= 16 and distance[i] <= 2 * iifds.uavR:
                rewardsum += (distance[i] - (2 * iifds.uavR)) / (2 * iifds.uavR) - 1
    else:
        for i in range(dis_len):  # 威胁区
            if i < 16 and distance[i] <= (iifds.uavR + iifds.cylinderR + 0.4):
                rewardsum += (distance[i] - (iifds.uavR + iifds.cylinderR + 0.4)) / (
                            iifds.uavR + iifds.cylinderR + 0.4) - 0.3
            elif i >= 16 and distance[i] <= 2 * iifds.uavR + 0.4:
                rewardsum += (distance[i] - (2 * iifds.uavR + 0.4)) / (2 * iifds.uavR + 0.4) - 0.3
        distancegoal = []
        for i in range(8):
            distancegoal.append(iifds.distanceCost(qNext[i], goal[i]))
        if ((distancegoal[0] > iifds.threshold) or (distancegoal[1] > iifds.threshold) or (
                distancegoal[2] > iifds.threshold)
                or (distancegoal[3] > iifds.threshold) or (distancegoal[4] > iifds.threshold) or (
                        distancegoal[5] > iifds.threshold)
                or (distancegoal[6] > iifds.threshold) or (distancegoal[7] > iifds.threshold)):
            for i in range(8):
                rewarduav.append(-distancegoal[i] / distances_g[i])
        else:
            for i in range(8):
                rewarduav.append(-distancegoal[i] / distances_g[i] + 3)

        rewardsum += sum(rewarduav)

        """奖励径直轨迹"""
        # q2qNext = qNext - q
        # q2goal = iifds.goal - q
        # theta = np.arccos(np.clip(q2qNext.dot(q2goal)/iifds.calVecLen(q2qNext)/iifds.calVecLen(q2goal),-1,1))
        # reward += -abs(theta) / (2*np.pi) * 0.2

        # if qBefore[0] is not None:
        #     x1, gam1, xres, gamres, _ = iifds.kinematicConstrant(q, qBefore, qNext)
        #     xDot = np.abs(x1 - xres)
        #     gamDot = np.abs(gam1 - gamres)
        #     reward += (- xDot / iifds.xmax - gamDot / iifds.gammax) * 0.5

    return rewardsum


def get_reward_multiple(env, qNext, dic):
    """多动态障碍环境获取reward函数"""
    reward = 0
    distance = env.distanceCost(qNext, dic['obsCenter'])
    if distance <= dic['obs_r']:
        reward += (distance - dic['obs_r']) / dic['obs_r'] - 1
    else:
        if distance < dic['obs_r'] + 0.4:
            tempR = dic['obs_r'] + 0.4
            reward += (distance - tempR) / tempR - 0.3
        distance1 = env.distanceCost(qNext, env.goal)
        distance2 = env.distanceCost(env.start, env.goal)
        if distance1 > env.threshold:
            reward += -distance1 / distance2
        else:
            reward += -distance1 / distance2 + 3
    return reward


def drawActionCurve(actionCurveList):
    """
    :param actionCurveList: 动作值列表
    :return: None 绘制图像
    """
    plt.figure()
    for i in range(actionCurveList.shape[1]):
        array = actionCurveList[:, i]
        if i == 0: label = 'row01'
        if i == 1: label = 'sigma01'
        if i == 2: label = 'theta1'
        if i == 3: label = 'row02'
        if i == 4: label = 'sigma02'
        if i == 5: label = 'theta2'
        plt.plot(np.arange(array.shape[0]), array, linewidth=2, label=label)
    plt.title('Variation diagram')
    plt.grid()
    plt.xlabel('time')
    plt.ylabel('value')
    plt.legend(loc='best')


def checkPath(apf):
    sum = 0
    for i in range(apf.path.shape[0] - 1):
        sum += apf.distanceCost(apf.path[i, :], apf.path[i + 1, :])
    for i, j in zip(apf.path, apf.dynamicSphere_Path):
        if apf.distanceCost(i, j) <= apf.dynamicSphere_R:
            print('与障碍物有交点，轨迹距离为：', sum)
            return
    print('与障碍物无交点，轨迹距离为：', sum)


def transformAction(actionBefore, actionBound, actionDim):
    """将强化学习输出的动作映射到指定的动作范围"""
    actionAfter = []
    for i in range(actionDim):
        action_i = actionBefore[i]
        action_bound_i = actionBound[i]
        actionAfter.append((action_i + 1) / 2 * (action_bound_i[1] - action_bound_i[0]) + action_bound_i[0])
    return actionAfter


def test(iifds, pi, conf):
    """动态单障碍环境测试训练效果"""
    iifds.reset()  # 重置环境
    start = iifds.start
    q = []
    qBefore = []
    goal = iifds.goal
    for i in range(8):
        q.append(iifds.start[i] + np.random.random(3) * 3)
        qBefore.append([None, None, None])
    rewardSum = 0
    for i in range(500):
        dic1, dic2 = iifds.updateObs()
        vObs1, obsCenter1, obsCenterNext1 = dic1['v'], dic1['obsCenter'], dic1['obsCenterNext']
        vObs2, obsCenter2, obsCenterNext2 = dic2['v'], dic2['obsCenter'], dic2['obsCenterNext']
        obsCenter = [obsCenter1, obsCenter2]
        vObs = [vObs1, vObs2]
        obsCenterNext = [obsCenterNext1, obsCenterNext2]
        obs_mix = iifds.calDynamicState(q, obsCenter)
        obs = np.array([])  # 中心控制器接受所有状态集合
        for k in range(len(obs_mix)):
            obs = np.hstack((obs, obs_mix[k]))  # 拼接状态为一个1*n向量
        obs = torch.as_tensor(obs, dtype=torch.float, device=device)
        action = pi(obs).cpu().detach().numpy()
        action = transformAction(action, conf.actionBound, conf.act_dim)
        # 与环境交互
        qNext = []
        for j in range(8):
            qNext.append(
                iifds.getqNext(j, q, obsCenter, vObs, action[j], action[3 * j + 1], action[3 * j + 2],
                               qBefore, goal))
        reward = getReward(obsCenterNext, qNext, q, qBefore, goal, iifds)
        rewardSum += reward

        qBefore = q
        q = qNext

        if ((iifds.distanceCost(goal[0], qNext[0]) < iifds.threshold) and (iifds.distanceCost(goal[1],
                                                                                              qNext[
                                                                                                  1]) < iifds.threshold) and (
                iifds.distanceCost(goal[2], qNext[2]) < iifds.threshold) and \
                (iifds.distanceCost(goal[3], qNext[3]) < iifds.threshold) and (
                        iifds.distanceCost(goal[4], qNext[4]) < iifds.threshold) \
                and (iifds.distanceCost(goal[5], qNext[5]) < iifds.threshold) and (
                        iifds.distanceCost(goal[6], qNext[6]) < iifds.threshold) \
                and (iifds.distanceCost(goal[7], qNext[7]) < iifds.threshold)):
            break
    return rewardSum


def setup_seed(seed):
    """设置随机数种子函数"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
