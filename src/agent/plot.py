import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors


def analysis_csv(input_csv, output_csv):

    # 从CSV文件读取数据
    df = pd.read_csv(input_csv, header=None,
                     names=["action", "agent_position", "target_position", "total_reward", "elapsed_time"])

    # 转换 'elapsed_time' 列为数值类型
    df['elapsed_time'] = pd.to_numeric(df['elapsed_time'], errors='coerce')
    df['total_reward'] = pd.to_numeric(df['total_reward'], errors='coerce')

    # 删除转换后出现 NaN 值的行
    df = df.dropna(subset=['elapsed_time'])

    # 找到 elapsed_time 为负数的索引（表示任务结束）
    end_indices = df[df['elapsed_time'].diff() < 0].index.tolist()

    # 提取每个任务的相关信息
    tasks = []
    for i in range(len(end_indices)):
        start_index = 0 if i == 0 else end_indices[i - 1] + 1
        end_index = end_indices[i]

        task_data = df.iloc[start_index:end_index - 1]

        total_rows = len(task_data)
        elapsed_time = task_data['elapsed_time'].iloc[-1]  # 取最后一行的值即可
        reward = task_data['total_reward'].iloc[-1]  # 取最后一行的值即可
        total_elapsed_time = task_data['elapsed_time'].sum()
        total_reward = task_data['total_reward'].sum()
        tasks.append([total_rows, total_elapsed_time, total_reward, elapsed_time, reward])

    # 创建DataFrame保存结果
    result_df = pd.DataFrame(tasks,
                             columns=['total_rows', 'total_elapsed_time', 'total_reward', 'elapsed_time', 'reward'])

    # 将结果保存为CSV文件
    result_df.to_csv(output_csv, index=False)


# Specify a font that supports a wide range of Unicode characters
plt.rcParams['font.sans-serif'] = ['SimHei']  # SimHei is a Chinese font, you can choose a different one if needed
plt.rcParams['axes.unicode_minus'] = False  # This is to avoid minus sign display issues in some fonts


def plot_task_analysis(dataframe):

    num_tasks = len(dataframe)
    colors = plt.cm.rainbow(np.linspace(0, 1, num_tasks))
    # 1. 利用彩虹色绘制总奖励
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(dataframe.index.astype(int) + 1, dataframe['total_reward'] / dataframe['total_rows'], marker='o',
            linestyle='-',
            color=colors[1], label='Mean', linewidth=2)
    ax.plot(dataframe.index.astype(int) + 1, dataframe['reward'], marker='o', linestyle='-', color=colors[2],
            label='Max', linewidth=2)
    ax.set_xlabel('Task')
    ax.set_ylabel('Total_reward')
    ax.legend(fontsize='large', frameon=False)  # 显示图例，并调整字体大小和去除边框

    # 2. 绘制平均开销时间和实际开销时间
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(dataframe.index.astype(int) + 1, dataframe['total_elapsed_time'] / dataframe['total_rows'], marker='o',
            linestyle='-',
            color=colors[1], label='Mean', linewidth=2)
    ax.plot(dataframe.index.astype(int) + 1, dataframe['elapsed_time'], marker='o', linestyle='-', color=colors[2],
            label='Max', linewidth=2)
    ax.set_xlabel('Task')
    ax.set_ylabel('Elapsed_time')
    ax.legend(fontsize='large', frameon=False)  # 显示图例，并调整字体大小和去除边框

    # Adjust label font size
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label]):
        item.set_fontsize(20)

    plt.show()


# 从CSV文件读取数据
df = pd.read_csv('task_analysis_result.csv')

# 调用函数进行绘图
plot_task_analysis(df)
