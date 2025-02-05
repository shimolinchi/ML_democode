import numpy as np
import matplotlib.pyplot as plt

# 参数设置
n_arms = 10  # 10臂赌博机
n_steps = 10000  # 实验步数
n_runs = 100  # 实验重复次数
epsilon = 0.1  # ε-贪婪策略参数
alpha = 0.1  # 增量计算方法的步长参数
sigma = 0.01  # 随机游走的标准差

# 初始化
def initialize():
    q_true = np.zeros(n_arms)  # 真实的动作值
    q_est_sample = np.zeros(n_arms)  # 样本平均方法的估计值
    q_est_incremental = np.zeros(n_arms)  # 增量计算方法的估计值
    action_counts_sample = np.zeros(n_arms)  # 样本平均方法的动作选择次数
    action_counts_incremental = np.zeros(n_arms)  # 增量计算方法的动作选择次数
    return q_true, q_est_sample, q_est_incremental, action_counts_sample, action_counts_incremental

# ε-贪婪策略
def epsilon_greedy(q_est, epsilon):
    if np.random.rand() < epsilon:
        return np.random.randint(n_arms)  # 随机选择动作
    else:
        return np.argmax(q_est)  # 选择当前估计值最大的动作

# 运行实验
def run_experiment():
    rewards_sample = np.zeros(n_steps)  # 样本平均方法的奖励
    rewards_incremental = np.zeros(n_steps)  # 增量计算方法的奖励
    optimal_sample = np.zeros(n_steps)  # 样本平均方法选择最优动作的次数
    optimal_incremental = np.zeros(n_steps)  # 增量计算方法选择最优动作的次数

    for run in range(n_runs):
        q_true, q_est_sample, q_est_incremental, action_counts_sample, action_counts_incremental = initialize()

        for step in range(n_steps):
            # 随机游走更新真实动作值
            q_true += np.random.normal(0, sigma, n_arms)

            # 样本平均方法
            action_sample = epsilon_greedy(q_est_sample, epsilon)
            # reward_sample = np.random.normal(q_true[action_sample], 0.3)
            reward_sample = q_true[action_sample]
            action_counts_sample[action_sample] += 1
            q_est_sample[action_sample] += (reward_sample - q_est_sample[action_sample]) / action_counts_sample[action_sample]
            rewards_sample[step] += reward_sample
            if action_sample == np.argmax(q_true):
                optimal_sample[step] += 1

            # 增量计算方法
            action_incremental = epsilon_greedy(q_est_incremental, epsilon)
            # reward_incremental = np.random.normal(q_true[action_incremental], 0.3)
            reward_incremental = q_true[action_incremental]
            q_est_incremental[action_incremental] += alpha * (reward_incremental - q_est_incremental[action_incremental])
            rewards_incremental[step] += reward_incremental
            if action_incremental == np.argmax(q_true):
                optimal_incremental[step] += 1

    # 平均结果
    rewards_sample /= n_runs
    rewards_incremental /= n_runs
    optimal_sample = optimal_sample / n_runs * 100
    optimal_incremental = optimal_incremental / n_runs * 100

    return rewards_sample, rewards_incremental, optimal_sample, optimal_incremental

# 运行实验并绘图
rewards_sample, rewards_incremental, optimal_sample, optimal_incremental = run_experiment()

plt.figure(figsize=(12, 6))

# 绘制平均奖励
plt.subplot(1, 2, 1)
plt.plot(rewards_sample, label='Sample Average')
plt.plot(rewards_incremental, label='Incremental (α=0.1)')
plt.xlabel('Steps')
plt.ylabel('Average Reward')
plt.legend()

# 绘制最优动作选择百分比
plt.subplot(1, 2, 2)
plt.plot(optimal_sample, label='Sample Average')
plt.plot(optimal_incremental, label='Incremental (α=0.1)')
plt.xlabel('Steps')
plt.ylabel('% Optimal Action')
plt.legend()

plt.tight_layout()
plt.show()