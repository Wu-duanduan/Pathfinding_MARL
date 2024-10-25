#!/usr/bin/python

import numpy as np
from dynamic_obstacle_environment import obs_list
from Method import getReward
import math
import heapq


class IIFDS:
    """使用IIFDS类训练时每次必须reset"""

    def __init__(self):
        """基本参数："""
        self.V0 = 1
        self.threshold = 0.5
        self.stepSize = 0.1
        self.lam = 8  # 越大考虑障碍物速度越明显
        self.numberofuav = 8
        # self.cylinderR = 0.5
        self.uavR = 0.2
        self.cylinderR = 0.5
        self.cylinderH = 4
        # 场景1
        startpoint = [np.random.randint(-4, 5, size=4), np.random.randint(-4, 5, size=4),
                      np.random.randint(1, 4, size=4)]
        endpoint1 = [np.random.randint(12, 13, size=4), np.random.randint(-12, 13, size=4),
                     np.random.randint(1, 4, size=1)]
        endpoint2 = [np.random.randint(-12, 13, size=1), np.random.randint(12, 13, size=1),
                     np.random.randint(1, 4, size=1)]
        endpoint3 = [np.random.randint(-12, -11, size=1), np.random.randint(-12, 13, size=1),
                     np.random.randint(1, 4, size=1)]
        endpoint4 = [np.random.randint(-12, 13, size=1), np.random.randint(-12, -11, size=1),
                     np.random.randint(1, 4, size=1)]
        self.start1 = np.array([startpoint[0][0], startpoint[1][0], startpoint[2][0]], dtype=float)
        self.start2 = np.array([startpoint[0][1], startpoint[1][1], startpoint[2][1]], dtype=float)
        self.start3 = np.array([startpoint[0][2], startpoint[1][2], startpoint[2][2]], dtype=float)
        self.start4 = np.array([startpoint[0][3], startpoint[1][3], startpoint[2][3]], dtype=float)
        self.goal1 = np.array([endpoint1[0][0], endpoint1[1][0], endpoint1[2][0]], dtype=float)
        self.goal2 = np.array([endpoint2[0][0], endpoint2[1][0], endpoint2[2][0]], dtype=float)
        self.goal3 = np.array([endpoint3[0][0], endpoint3[1][0], endpoint3[2][0]], dtype=float)
        self.goal4 = np.array([endpoint4[0][0], endpoint4[1][0], endpoint4[2][0]], dtype=float)
        self.start5 = self.goal1
        self.start6 = self.goal2
        self.start7 = self.goal3
        self.start8 = self.goal4
        self.goal5 = self.start1
        self.goal6 = self.start2
        self.goal7 = self.start3
        self.goal8 = self.start4

        # 场景2
        # startpoint = [np.random.randint(-4, 5, size=8), np.random.randint(-4, 5, size=8), np.random.randint(1, 4, size=8)]
        # endpoint12 = [np.random.randint(12, 16, size=2), np.random.randint(-15, 16, size=2), np.random.randint(1, 4, size=2)]
        # endpoint34 = [np.random.randint(-15, 16, size=2), np.random.randint(12, 16, size=2), np.random.randint(1, 4, size=2)]
        # endpoint56 = [np.random.randint(-15, -11, size=2), np.random.randint(-15, 16, size=2), np.random.randint(1, 4, size=2)]
        # endpoint78 = [np.random.randint(-15, 16, size=2), np.random.randint(-15, -11, size=2), np.random.randint(1, 4, size=2)]
        # self.start1 = np.array([startpoint[0][0], startpoint[1][0], startpoint[2][0]], dtype=float)
        # self.start2 = np.array([startpoint[0][1], startpoint[1][1], startpoint[2][1]], dtype=float)
        # self.start3 = np.array([startpoint[0][2], startpoint[1][2], startpoint[2][2]], dtype=float)
        # self.start4 = np.array([startpoint[0][3], startpoint[1][3], startpoint[2][3]], dtype=float)
        # self.start5 = np.array([startpoint[0][4], startpoint[1][4], startpoint[2][4]], dtype=float)
        # self.start6 = np.array([startpoint[0][5], startpoint[1][5], startpoint[2][5]], dtype=float)
        # self.start7 = np.array([startpoint[0][6], startpoint[1][6], startpoint[2][6]], dtype=float)
        # self.start8 = np.array([startpoint[0][7], startpoint[1][7], startpoint[2][7]], dtype=float)
        # self.goal1 = np.array([endpoint12[0][0], endpoint12[1][0], endpoint12[2][0]], dtype=float)
        # self.goal2 = np.array([endpoint12[0][1], endpoint12[1][1], endpoint12[2][1]], dtype=float)
        # self.goal3 = np.array([endpoint34[0][0], endpoint34[1][0], endpoint34[2][0]], dtype=float)
        # self.goal4 = np.array([endpoint34[0][1], endpoint34[1][1], endpoint34[2][1]], dtype=float)
        # self.goal5 = np.array([endpoint56[0][0], endpoint56[1][0], endpoint56[2][0]], dtype=float)
        # self.goal6 = np.array([endpoint56[0][1], endpoint56[1][1], endpoint56[2][1]], dtype=float)
        # self.goal7 = np.array([endpoint78[0][0], endpoint78[1][0], endpoint78[2][0]], dtype=float)
        # self.goal8 = np.array([endpoint78[0][1], endpoint78[1][1], endpoint78[2][1]], dtype=float)

        self.start = [self.start1, self.start2, self.start3, self.start4, self.start5, self.start6, self.start7,
                      self.start8]
        self.goal = [self.goal1, self.goal2, self.goal3, self.goal4, self.goal5, self.goal6, self.goal7, self.goal8]

        self.timelog = 0  # 时间，用来计算动态障碍的位置
        self.timeStep = 0.1

        self.xmax = 10 / 180 * np.pi  # 偏航角速度最大值  每个步长允许变化的角度
        self.gammax = 10 / 180 * np.pi  # 爬升角速度最大值  每个步长允许变化的角度
        self.maximumClimbingAngle = 100 / 180 * np.pi  # 最大爬升角
        self.maximumSubductionAngle = - 75 / 180 * np.pi  # 最大俯冲角

        self.vObs1 = None
        self.vObsNext1 = None
        self.vObs2 = None
        self.vObsNext2 = None

        self.path1 = np.array([[]]).reshape(-1, 3)  # 保存动态球的运动轨迹
        self.path2 = np.array([[]]).reshape(-1, 3)  # 保存动态球的运动轨迹

        self.env_num = len(obs_list)
        self.env1 = obs_list[0]
        self.env2 = obs_list[2]

    def reset(self):
        self.timelog = 0  # 重置时间
        self.path1 = np.array([[]]).reshape(-1, 3)  # 清空障碍路径记录表
        self.path2 = np.array([[]]).reshape(-1, 3)  # 清空障碍路径记录表
        a, b = np.random.randint(0, self.env_num, size=2)
        self.env1 = obs_list[a]  # 随机一个训练环境
        self.env2 = obs_list[b]  # 随机一个训练环境

    def updateObs(self, if_test=False):
        """返回位置与速度。"""
        if if_test:
            """测试环境"""
            self.timelog, dic1 = obs_list[0](self.timelog, self.timeStep)
            self.timelog, dic2 = obs_list[2](self.timelog, self.timeStep)
        else:
            """否则用reset时随机的一个环境"""
            self.timelog, dic1 = self.env1(self.timelog, self.timeStep)
            self.timelog, dic2 = self.env2(self.timelog, self.timeStep)
        self.vObs1 = dic1['v']
        self.path1 = np.vstack((self.path1, dic1['obsCenter']))
        self.vObs2 = dic2['v']
        self.path2 = np.vstack((self.path2, dic2['obsCenter']))
        return dic1, dic2

    def calDynamicState(self, uavPos, obsCenter):
        """强化学习模型获得的state。"""
        dic = {'uav1': [], 'uav2': [], 'uav3': [], 'uav4': [], 'uav5': [], 'uav6': [], 'uav7': [], 'uav8': []}
        vObs = [self.vObs1, self.vObs2]
        s = []
        for i in range(8):
            for j in range(2):
                s1 = (obsCenter[j] - uavPos[i]) * (
                        self.distanceCost(obsCenter[j], uavPos[i]) - (self.cylinderR + self.uavR)) / self.distanceCost(
                    obsCenter[j],
                    uavPos[i])
                s.append(s1)
                s2 = vObs[j]
                s.append(s2)
            s3 = self.goal[i] - uavPos[i]
            s.append(s3)
        # 不仅考虑到观测障碍物 额外还能观测最近的两架无人机
        distance = np.ones([8, 8]) * 100
        for i in range(8):
            for j in range(8):
                if i != j:
                    distance[i][j] = self.distanceCost(uavPos[i], uavPos[j])
        # print(distance[0][np.argpartition(distance[0],3)[:3]])
        z = []
        self.num_com = 2
        self.uav_com = np.zeros([8, self.num_com])
        self.index_com = np.zeros([8, self.num_com])
        for i in range(8):
            self.uav_com[i] = heapq.nsmallest(self.num_com, distance[i])
            self.index_com[i] = list(map(distance[i].tolist().index, self.uav_com[i]))
            # print(index_com[i][0])
            for j in range(int(self.num_com)):
                # z1 = int(self.index_com[i][j])
                z1 = (uavPos[int(self.index_com[i][j])] - uavPos[i]) * (
                        self.distanceCost(uavPos[int(self.index_com[i][j])],
                                          uavPos[i]) - 2 * self.uavR) / self.distanceCost(
                    uavPos[int(self.index_com[i][j])],
                    uavPos[i])
                z.append(z1)
        dic['uav1'].append(np.hstack((s[0], s[1], s[2], s[3], s[4], z[0], z[1])))
        dic['uav2'].append(np.hstack((s[5], s[6], s[7], s[8], s[9], z[2], z[3])))
        dic['uav3'].append(np.hstack((s[10], s[11], s[12], s[13], s[14], z[4], z[5])))
        dic['uav4'].append(np.hstack((s[15], s[16], s[17], s[18], s[19], z[6], z[7])))
        dic['uav5'].append(np.hstack((s[20], s[21], s[22], s[23], s[24], z[8], z[9])))
        dic['uav6'].append(np.hstack((s[25], s[26], s[27], s[28], s[29], z[10], z[11])))
        dic['uav7'].append(np.hstack((s[30], s[31], s[32], s[33], s[34], z[12], z[13])))
        dic['uav8'].append(np.hstack((s[35], s[36], s[37], s[38], s[39], z[14], z[15])))
        return dic

    def calRepulsiveMatrix(self, uavPos, obsCenter, cylinderR, row0, goal):
        n = self.partialDerivativeSphere(obsCenter, uavPos, cylinderR)
        tempD = self.distanceCost(uavPos, obsCenter) - cylinderR
        row = row0 * np.exp(1 - 1 / (self.distanceCost(uavPos, goal) * tempD))
        T = self.calculateT(obsCenter, uavPos, cylinderR)
        repulsiveMatrix = np.dot(-n, n.T) / T ** (1 / row) / np.dot(n.T, n)[0][0]
        return repulsiveMatrix

    def calTangentialMatrix(self, uavPos, obsCenter, cylinderR, theta, sigma0, goal):
        n = self.partialDerivativeSphere(obsCenter, uavPos, cylinderR)
        T = self.calculateT(obsCenter, uavPos, cylinderR)
        partialX = (uavPos[0] - obsCenter[0]) * 2 / cylinderR ** 2
        partialY = (uavPos[1] - obsCenter[1]) * 2 / cylinderR ** 2
        partialZ = (uavPos[2] - obsCenter[2]) * 2 / cylinderR ** 2
        tk1 = np.array([partialY, -partialX, 0], dtype=float).reshape(-1, 1)
        tk2 = np.array([partialX * partialZ, partialY * partialZ, -partialX ** 2 - partialY ** 2], dtype=float).reshape(
            -1, 1)
        originalPoint = np.array([np.cos(theta), np.sin(theta), 0]).reshape(1, -1)
        tk = self.trans(originalPoint, tk1.squeeze(), tk2.squeeze(), n.squeeze())
        tempD = self.distanceCost(uavPos, obsCenter) - cylinderR
        sigma = sigma0 * np.exp(1 - 1 / (self.distanceCost(uavPos, goal) * tempD))
        tangentialMatrix = tk.dot(n.T) / T ** (1 / sigma) / self.calVecLen(tk.squeeze()) / self.calVecLen(n.squeeze())
        return tangentialMatrix

    def calRepulsiveMatrix2(self, uavPos, obsCenter, cylinderR, row0, goal):
        n = self.partialDerivativeSphere2(obsCenter, uavPos, cylinderR)
        tempD = self.distanceCost(uavPos[0:2], obsCenter[0:2]) - cylinderR
        row = row0 * np.exp(1 - 1 / (self.distanceCost(uavPos, goal) * tempD))
        T = self.calculateT2(obsCenter, uavPos, cylinderR)
        repulsiveMatrix = np.dot(-n, n.T) / T ** (1 / row) / np.dot(n.T, n)[0][0]
        return repulsiveMatrix

    def calTangentialMatrix2(self, uavPos, obsCenter, cylinderR, theta, sigma0, goal):
        n = self.partialDerivativeSphere2(obsCenter, uavPos, cylinderR)
        T = self.calculateT2(obsCenter, uavPos, cylinderR)
        partialX = (uavPos[0] - obsCenter[0]) * 2 / cylinderR ** 2
        partialY = (uavPos[1] - obsCenter[1]) * 2 / cylinderR ** 2
        partialZ = 0
        tk1 = np.array([partialY, -partialX, 0], dtype=float).reshape(-1, 1)
        tk2 = np.array([partialX * partialZ, partialY * partialZ, -partialX ** 2 - partialY ** 2], dtype=float).reshape(
            -1, 1)
        originalPoint = np.array([np.cos(theta), np.sin(theta), 0]).reshape(1, -1)
        tk = self.trans(originalPoint, tk1.squeeze(), tk2.squeeze(), n.squeeze())
        tempD = self.distanceCost(uavPos[0:2], obsCenter[0:2]) - cylinderR
        sigma = sigma0 * np.exp(1 - 1 / (self.distanceCost(uavPos, goal) * tempD))
        tangentialMatrix = tk.dot(n.T) / T ** (1 / sigma) / self.calVecLen(tk.squeeze()) / self.calVecLen(n.squeeze())
        return tangentialMatrix

    def getqNext(self, i, q, obsCenter, vObs, row0, sigma0, theta, qBefore, goal):
        uavPos = q[i]
        goal = goal[i]
        qBefore = qBefore[i]
        u = self.initField(uavPos, self.V0, goal)
        repulsiveMatrix = self.calRepulsiveMatrix2(uavPos, obsCenter[0], self.cylinderR + self.uavR, row0, goal)
        repulsiveMatrix += self.calRepulsiveMatrix2(uavPos, obsCenter[1], self.cylinderR + self.uavR, row0, goal)

        tangentialMatrix = self.calTangentialMatrix2(uavPos, obsCenter[0], self.cylinderR + self.uavR, theta, sigma0,
                                                     goal)
        tangentialMatrix += self.calTangentialMatrix2(uavPos, obsCenter[1], self.cylinderR + self.uavR, theta, sigma0,
                                                      goal)
        for j in range(int(self.num_com)):
            repulsiveMatrix += self.calRepulsiveMatrix(uavPos, q[int(self.index_com[i][j])], 2 * self.uavR, row0, goal)
            tangentialMatrix += self.calTangentialMatrix(uavPos, q[int(self.index_com[i][j])], 2 * self.uavR, theta,
                                                         sigma0, goal)
        T1 = self.calculateT(obsCenter[0], uavPos, self.cylinderR + self.uavR)
        T2 = self.calculateT(obsCenter[1], uavPos, self.cylinderR + self.uavR)
        vp1 = np.exp(-T1 / self.lam) * vObs[0]
        vp2 = np.exp(-T2 / self.lam) * vObs[1]
        M = np.eye(3) + repulsiveMatrix + tangentialMatrix
        ubar = (M.dot(u - vp1.reshape(-1, 1)).T + vp1.reshape(1, -1)).squeeze() + (
                M.dot(u - vp2.reshape(-1, 1)).T + vp2.reshape(1, -1)).squeeze()
        # 限制ubar的模长，避免进入障碍内部后轨迹突变
        if self.calVecLen(ubar) > 5:
            ubar = ubar / self.calVecLen(ubar) * 5
        if qBefore[0] is None:
            uavNextPos = uavPos + ubar * self.stepSize
        else:
            uavNextPos = uavPos + ubar * self.stepSize
            _, _, _, _, qNext = self.kinematicConstrant(uavPos, qBefore, uavNextPos)
        return uavNextPos

    def kinematicConstrant(self, q, qBefore, qNext):
        """
        运动学约束函数 返回(上一时刻航迹角，上一时刻爬升角，约束后航迹角，约束后爬升角，约束后下一位置qNext)
        """
        # 计算qBefore到q航迹角x1,gam1
        qBefore2q = q - qBefore
        if qBefore2q[0] != 0 or qBefore2q[1] != 0:
            x1 = np.arcsin(
                np.abs(qBefore2q[1] / np.sqrt(qBefore2q[0] ** 2 + qBefore2q[1] ** 2)))  # 这里计算的角限定在了第一象限的角 0-pi/2
            gam1 = np.arcsin(qBefore2q[2] / np.sqrt(np.sum(qBefore2q ** 2)))
        else:
            return None, None, None, None, qNext
        # 计算q到qNext航迹角x2,gam2
        q2qNext = qNext - q
        x2 = np.arcsin(np.abs(q2qNext[1] / np.sqrt(q2qNext[0] ** 2 + q2qNext[1] ** 2)))  # 这里同理计算第一象限的角度
        gam2 = np.arcsin(q2qNext[2] / np.sqrt(np.sum(q2qNext ** 2)))

        # 根据不同象限计算矢量相对于x正半轴的角度 0-2 * pi
        if qBefore2q[0] > 0 and qBefore2q[1] > 0:
            x1 = x1
        if qBefore2q[0] < 0 and qBefore2q[1] > 0:
            x1 = np.pi - x1
        if qBefore2q[0] < 0 and qBefore2q[1] < 0:
            x1 = np.pi + x1
        if qBefore2q[0] > 0 and qBefore2q[1] < 0:
            x1 = 2 * np.pi - x1
        if qBefore2q[0] > 0 and qBefore2q[1] == 0:
            x1 = 0
        if qBefore2q[0] == 0 and qBefore2q[1] > 0:
            x1 = np.pi / 2
        if qBefore2q[0] < 0 and qBefore2q[1] == 0:
            x1 = np.pi
        if qBefore2q[0] == 0 and qBefore2q[1] < 0:
            x1 = np.pi * 3 / 2

        # 根据不同象限计算与x正半轴的角度
        if q2qNext[0] > 0 and q2qNext[1] > 0:
            x2 = x2
        if q2qNext[0] < 0 and q2qNext[1] > 0:
            x2 = np.pi - x2
        if q2qNext[0] < 0 and q2qNext[1] < 0:
            x2 = np.pi + x2
        if q2qNext[0] > 0 and q2qNext[1] < 0:
            x2 = 2 * np.pi - x2
        if q2qNext[0] > 0 and q2qNext[1] == 0:
            x2 = 0
        if q2qNext[0] == 0 and q2qNext[1] > 0:
            x2 = np.pi / 2
        if q2qNext[0] < 0 and q2qNext[1] == 0:
            x2 = np.pi
        if q2qNext[0] == 0 and q2qNext[1] < 0:
            x2 = np.pi * 3 / 2

        # 约束航迹角x   xres为约束后的航迹角
        deltax1x2 = self.angleVec(q2qNext[0:2], qBefore2q[0:2])  # 利用点乘除以模长乘积求xoy平面投影的夹角
        if deltax1x2 < self.xmax:
            xres = x2
        elif x1 - x2 > 0 and x1 - x2 < np.pi:  # 注意这几个逻辑
            xres = x1 - self.xmax
        elif x1 - x2 > 0 and x1 - x2 > np.pi:
            xres = x1 + self.xmax
        elif x1 - x2 < 0 and x2 - x1 < np.pi:
            xres = x1 + self.xmax
        else:
            xres = x1 - self.xmax

        # 约束爬升角gam   注意：爬升角只用讨论在-pi/2到pi/2区间，这恰好与arcsin的值域相同。  gamres为约束后的爬升角
        if np.abs(gam1 - gam2) <= self.gammax:
            gamres = gam2
        elif gam2 > gam1:
            gamres = gam1 + self.gammax
        else:
            gamres = gam1 - self.gammax
        if gamres > self.maximumClimbingAngle:
            gamres = self.maximumClimbingAngle
        if gamres < self.maximumSubductionAngle:
            gamres = self.maximumSubductionAngle

        # 计算约束过后下一个点qNext的坐标
        Rq2qNext = self.distanceCost(q, qNext)
        deltax = Rq2qNext * np.cos(gamres) * np.cos(xres)
        deltay = Rq2qNext * np.cos(gamres) * np.sin(xres)
        deltaz = Rq2qNext * np.sin(gamres)

        qNext = q + np.array([deltax, deltay, deltaz])
        return x1, gam1, xres, gamres, qNext

    def loop(self):
        uavPos = self.start
        row0 = 0.5
        theta = 0.5
        sigma0 = 0.5
        path = self.start.reshape(1, -1)
        qBefore = [None, None, None]
        reward = 0
        for i in range(500):
            dic = self.updateObs(if_test=True)
            vObs, obsCenter = dic['v'], dic['obsCenter']
            uavNextPos = self.getqNext(uavPos, obsCenter, vObs, row0, sigma0, theta, qBefore)
            reward += getReward(obsCenter, uavNextPos, uavPos, qBefore, self)
            qBefore = uavPos
            uavPos = uavNextPos
            if self.distanceCost(uavPos, self.goal) < self.threshold:
                path = np.vstack((path, self.goal))
                _ = iifds.updateObs(if_test=True)
                break
            path = np.vstack((path, uavPos))
        print('路径的长度为:%f' % self.calPathLen(path))
        print('奖励为:%f' % reward)
        np.savetxt('./data_csv/pathMatrix.csv', path, delimiter=',')
        self.save_data()

    @staticmethod
    def distanceCost(point1, point2):
        return np.sqrt(np.sum((point1 - point2) ** 2))

    def initField(self, pos, V0, goal):
        """计算初始流场，返回列向量。"""
        temp1 = pos[0] - goal[0]
        temp2 = pos[1] - goal[1]
        temp3 = pos[2] - goal[2]
        temp4 = self.distanceCost(pos, goal)
        return -np.array([temp1, temp2, temp3], dtype=float).reshape(-1, 1) * V0 / temp4

    @staticmethod
    def partialDerivativeSphere(obs, pos, r):
        """计算球障碍物方程偏导数，返回列向量。"""
        temp1 = pos[0] - obs[0]
        temp2 = pos[1] - obs[1]
        temp3 = pos[2] - obs[2]
        return np.array([temp1, temp2, temp3], dtype=float).reshape(-1, 1) * 2 / r ** 2

    @staticmethod
    def calculateT(obs, pos, r):
        """计算T。"""
        temp1 = pos[0] - obs[0]
        temp2 = pos[1] - obs[1]
        temp3 = pos[2] - obs[2]
        return (temp1 ** 2 + temp2 ** 2 + temp3 ** 2) / r ** 2

    @staticmethod
    def partialDerivativeSphere2(obs, pos, r):
        """计算球障碍物方程偏导数，返回列向量。"""
        temp1 = pos[0] - obs[0]
        temp2 = pos[1] - obs[1]
        temp3 = 0
        return np.array([temp1, temp2, temp3], dtype=float).reshape(-1, 1) * 2 / r ** 2

    @staticmethod
    def calculateT2(obs, pos, r):
        """计算T。"""
        temp1 = pos[0] - obs[0]
        temp2 = pos[1] - obs[1]
        temp3 = 0
        return (temp1 ** 2 + temp2 ** 2 + temp3 ** 3) / r ** 2

    def calPathLen(self, path):
        """计算一个轨迹的长度。"""
        num = path.shape[0]
        len = 0
        for i in range(num - 1):
            len += self.distanceCost(path[i, :], path[i + 1, :])
        return len

    def trans(self, originalPoint, xNew, yNew, zNew):
        """
        坐标变换后地球坐标下坐标
        newX, newY, newZ是新坐标下三个轴上的方向向量
        返回列向量
        """
        lenx = self.calVecLen(xNew)
        cosa1 = xNew[0] / lenx
        cosb1 = xNew[1] / lenx
        cosc1 = xNew[2] / lenx

        leny = self.calVecLen(yNew)
        cosa2 = yNew[0] / leny
        cosb2 = yNew[1] / leny
        cosc2 = yNew[2] / leny

        lenz = self.calVecLen(zNew)
        cosa3 = zNew[0] / lenz
        cosb3 = zNew[1] / lenz
        cosc3 = zNew[2] / lenz

        B = np.array([[cosa1, cosb1, cosc1],
                      [cosa2, cosb2, cosc2],
                      [cosa3, cosb3, cosc3]], dtype=float)

        invB = np.linalg.inv(B)
        return np.dot(invB, originalPoint.T)

    @staticmethod
    def calVecLen(vec):
        """计算向量模长。"""
        return np.sqrt(np.sum(vec ** 2))

    @staticmethod
    def angleVec(vec1, vec2):  # 计算两个向量之间的夹角
        temp = np.dot(vec1, vec2) / np.sqrt(np.sum(vec1 ** 2)) / np.sqrt(np.sum(vec2 ** 2))
        temp = np.clip(temp, -1, 1)  # 可能存在精度误差导致上一步的temp略大于1，因此clip
        theta = np.arccos(temp)
        return theta

    def save_data(self):
        np.savetxt('./data_csv/start1.csv', self.start1, delimiter=',')
        np.savetxt('./data_csv/goal1.csv', self.goal1, delimiter=',')
        np.savetxt('./data_csv/start2.csv', self.start2, delimiter=',')
        np.savetxt('./data_csv/goal2.csv', self.goal2, delimiter=',')
        np.savetxt('./data_csv/start3.csv', self.start3, delimiter=',')
        np.savetxt('./data_csv/goal3.csv', self.goal3, delimiter=',')
        np.savetxt('./data_csv/start4.csv', self.start4, delimiter=',')
        np.savetxt('./data_csv/goal4.csv', self.goal4, delimiter=',')
        np.savetxt('./data_csv/start5.csv', self.start5, delimiter=',')
        np.savetxt('./data_csv/goal5.csv', self.goal5, delimiter=',')
        np.savetxt('./data_csv/start6.csv', self.start6, delimiter=',')
        np.savetxt('./data_csv/goal6.csv', self.goal6, delimiter=',')
        np.savetxt('./data_csv/start7.csv', self.start7, delimiter=',')
        np.savetxt('./data_csv/goal7.csv', self.goal7, delimiter=',')
        np.savetxt('./data_csv/start8.csv', self.start8, delimiter=',')
        np.savetxt('./data_csv/goal8.csv', self.goal8, delimiter=',')
        np.savetxt('./data_csv/cylinder_r.csv', np.array([self.cylinderR]), delimiter=',')
        np.savetxt('./data_csv/cylinder_h.csv', np.array([self.cylinderH]), delimiter=',')
        np.savetxt('./data_csv/obs_trace1.csv', self.path1, delimiter=',')
        np.savetxt('./data_csv/obs_trace2.csv', self.path2, delimiter=',')


if __name__ == "__main__":
    iifds = IIFDS()
    iifds.loop()
