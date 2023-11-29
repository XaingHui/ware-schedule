import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors


def merge_csv(input_csv, input_csv_random):
    # 读取两个 CSV 文件
    df1 = pd.read_csv(input_csv)
    df2 = pd.read_csv(input_csv_random)

    # 给第二个 CSV 文件的列名加上 "random" 后缀
    df2.columns = [col + '_random' for col in df2.columns]

    # 合并两个 DataFrame
    merged_df = pd.concat([df1, df2], axis=1)

    # 保存合并后的 DataFrame 到新的 CSV 文件
    merged_df.to_csv('task_analysis.csv', index=False)


# Specify a font that supports a wide range of Unicode characters
plt.rcParams['font.sans-serif'] = ['SimHei']  # SimHei is a Chinese font, you can choose a different one if needed
plt.rcParams['axes.unicode_minus'] = False  # This is to avoid minus sign display issues in some fonts


def plot_task_analysis(dataframe):
    num_tasks = len(dataframe)
    colors = plt.cm.rainbow(np.linspace(0, 1, num_tasks))
    # 1. 利用彩虹色绘制总奖励
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(dataframe.index.astype(int) + 1, dataframe['reward'], marker='o',
            linestyle='-',
            color=colors[1], label='reward_real', linewidth=2)
    ax.plot(dataframe.index.astype(int) + 1, dataframe['reward_random'], marker='o', linestyle='-', color=colors[2],
            label='reward_random', linewidth=2)
    ax.set_xlabel('Task')
    ax.set_ylabel('Max_reward')
    ax.legend(fontsize='large', frameon=False)  # 显示图例，并调整字体大小和去除边框

    # 2. 绘制平均奖励
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(dataframe.index.astype(int) + 1, dataframe['total_reward'] / dataframe['total_rows'], marker='o',
            linestyle='-',
            color=colors[1], label='reward_real', linewidth=2)
    ax.plot(dataframe.index.astype(int) + 1, dataframe['total_reward_random'] / dataframe['total_rows_random'], marker='o', linestyle='-', color=colors[2],
            label='reward_random', linewidth=2)
    ax.set_xlabel('Task')
    ax.set_ylabel('Mean_reward')
    ax.legend(fontsize='large', frameon=False)  # 显示图例，并调整字体大小和去除边框


    # 1. 利用彩虹色绘制总开销时间
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(dataframe.index.astype(int) + 1, dataframe['elapsed_time'], marker='o',
            linestyle='-',
            color=colors[1], label='elapsed_time_real', linewidth=2)
    ax.plot(dataframe.index.astype(int) + 1, dataframe['elapsed_time_random'], marker='o', linestyle='-', color=colors[2],
            label='elapsed_time_random', linewidth=2)
    ax.set_xlabel('Task')
    ax.set_ylabel('Max_elapsed_time')
    ax.legend(fontsize='large', frameon=False)  # 显示图例，并调整字体大小和去除边框

    # 2. 绘制平均开销时间
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(dataframe.index.astype(int) + 1, dataframe['total_elapsed_time'] / dataframe['total_rows'], marker='o',
            linestyle='-',
            color=colors[1], label='elapsed_time_real', linewidth=2)
    ax.plot(dataframe.index.astype(int) + 1, dataframe['total_elapsed_time_random'] / dataframe['total_rows_random'], marker='o', linestyle='-', color=colors[2],
            label='elapsed_time_random', linewidth=2)
    ax.set_xlabel('Task')
    ax.set_ylabel('Mean_elapsed_time')
    ax.legend(fontsize='large', frameon=False)  # 显示图例，并调整字体大小和去除边框

    # 1. 利用彩虹色绘制总冲突
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(dataframe.index.astype(int) + 1, dataframe['conflict_count'], marker='o',
            linestyle='-',
            color=colors[1], label='conflict_count_real', linewidth=2)
    ax.plot(dataframe.index.astype(int) + 1, dataframe['conflict_count_random'], marker='o', linestyle='-',
            color=colors[2],
            label='conflict_count_random', linewidth=2)
    ax.set_xlabel('Task')
    ax.set_ylabel('Max_conflict_count')
    ax.legend(fontsize='large', frameon=False)  # 显示图例，并调整字体大小和去除边框

    # # 2. 绘制平均冲突
    # fig, ax = plt.subplots(figsize=(10, 6))
    # ax.plot(dataframe.index.astype(int) + 1, dataframe['total_conflict_count'] / dataframe['total_rows'], marker='o',
    #         linestyle='-',
    #         color=colors[1], label='conflict_count_real', linewidth=2)
    # ax.plot(dataframe.index.astype(int) + 1, dataframe['total_conflict_count_random'] / dataframe['total_rows_random'],
    #         marker='o', linestyle='-', color=colors[2],
    #         label='conflict_count_random', linewidth=2)
    # ax.set_xlabel('Task')
    # ax.set_ylabel('Mean_conflict_count')
    # ax.legend(fontsize='large', frameon=False)  # 显示图例，并调整字体大小和去除边框


    # Adjust label font size
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label]):
        item.set_fontsize(20)

    plt.show()


if __name__ == '__main__':
    merge_csv('./agent/task_analysis_result.csv', './agent_random_5/task_analysis_result_random.csv')
    df = pd.read_csv('task_analysis.csv')
    plot_task_analysis(df)
