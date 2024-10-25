#!/usr/bin/python

import torch
import numpy as np
import matplotlib.pyplot as plt
from IIFDS2 import IIFDS
from Method import checkPath, drawActionCurve, getReward, setup_seed
from arguments import parse_args
import random

seed = 18
np.random.seed(seed)
random.seed(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    iifds = IIFDS()
    arglist = parse_args()
    # 加载模型
    actors_cur = [None for _ in range(iifds.numberofuav)]
    for i in range(iifds.numberofuav):
        actors_cur[i] = torch.load('TrainedModel/Actor.%d.pkl' % i, map_location=device)

    start = iifds.start
    q = []
    qBefore = []
    goal = iifds.goal
    for i in range(8):
        q.append(iifds.start[i])
        qBefore.append([None, None, None])
    path1 = iifds.start1.reshape(1, -1)
    path2 = iifds.start2.reshape(1, -1)
    path3 = iifds.start3.reshape(1, -1)
    path4 = iifds.start4.reshape(1, -1)
    path5 = iifds.start5.reshape(1, -1)
    path6 = iifds.start6.reshape(1, -1)
    path7 = iifds.start7.reshape(1, -1)
    path8 = iifds.start8.reshape(1, -1)
    rewardSum = 0
    for i in range(500):
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
        action_n = [agent(torch.from_numpy(obs).to(arglist.device, torch.float)).detach().cpu().numpy() \
                    for agent, obs in zip(actors_cur, obs_n)]
        action_n = np.clip(action_n, arglist.action_limit_min, arglist.action_limit_max)
        action_n = action_n.reshape(-1)

        qNext = []
        for j in range(8):
            qNext.append(iifds.getqNext(j, q, obsCenter, vObs, action_n[3 * j], action_n[3 * j + 1],
                                        action_n[3 * j + 2], qBefore, goal))
        rew_n = [getReward(obsCenterNext, qNext, q, qBefore, goal, iifds) for _ in
                 range(iifds.numberofuav)]  # 每个agent使用相同的reward
        rewardSum += rew_n[0]

        qBefore = q
        q = qNext

        if ((iifds.distanceCost(goal[0], qNext[0]) < iifds.threshold) and (iifds.distanceCost(goal[1], qNext[1]) < iifds.threshold) and (
                iifds.distanceCost(goal[2], qNext[2]) < iifds.threshold) and \
                (iifds.distanceCost(goal[3], qNext[3]) < iifds.threshold) and (
                        iifds.distanceCost(goal[4], qNext[4]) < iifds.threshold) \
                and (iifds.distanceCost(goal[5], qNext[5]) < iifds.threshold) and (
                        iifds.distanceCost(goal[6], qNext[6]) < iifds.threshold) \
                and (iifds.distanceCost(goal[7], qNext[7]) < iifds.threshold)):
            path1 = np.vstack((path1, goal[0]))
            path2 = np.vstack((path2, goal[1]))
            path3 = np.vstack((path3, goal[2]))
            path4 = np.vstack((path4, goal[3]))
            path5 = np.vstack((path5, goal[4]))
            path6 = np.vstack((path6, goal[5]))
            path7 = np.vstack((path7, goal[6]))
            path8 = np.vstack((path8, goal[7]))
            _ = iifds.updateObs(if_test=True)
            break
        path1 = np.vstack((path1, q[0]))
        path2 = np.vstack((path2, q[1]))
        path3 = np.vstack((path3, q[2]))
        path4 = np.vstack((path4, q[3]))
        path5 = np.vstack((path5, q[4]))
        path6 = np.vstack((path6, q[5]))
        path7 = np.vstack((path7, q[6]))
        path8 = np.vstack((path8, q[7]))
    np.savetxt('./MADDPG_data_csv/pathMatrix1.csv', path1, delimiter=',')
    np.savetxt('./MADDPG_data_csv/pathMatrix2.csv', path2, delimiter=',')
    np.savetxt('./MADDPG_data_csv/pathMatrix3.csv', path3, delimiter=',')
    np.savetxt('./MADDPG_data_csv/pathMatrix4.csv', path4, delimiter=',')
    np.savetxt('./MADDPG_data_csv/pathMatrix5.csv', path5, delimiter=',')
    np.savetxt('./MADDPG_data_csv/pathMatrix6.csv', path6, delimiter=',')
    np.savetxt('./MADDPG_data_csv/pathMatrix7.csv', path7, delimiter=',')
    np.savetxt('./MADDPG_data_csv/pathMatrix8.csv', path8, delimiter=',')
    iifds.save_data()
    print('该路径的奖励总和为:%f' % rewardSum)
    plt.show()
