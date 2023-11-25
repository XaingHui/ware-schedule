import pandas as pd

# 读取CSV文件
df = pd.read_csv('simulation_records.csv')

# 初始化变量
tasks = []
current_task = {"total_reward": 0, "total_row": 0, "elapsed_time": 0}

# 遍历数据框的每一行
for index, row in df.iterrows():
    if row['total_reward'] == 0 and current_task["total_reward"] != 0:
        # 当出现动作为0且之前的任务的总奖励不为0时，表示一个任务结束
        tasks.append(current_task)
        current_task = {"total_reward": 0, "total_row": 0, "elapsed_time": 0}
    else:
        # 累加总奖励
        if row['total_reward'] != 0:
            current_task["total_row"] += 1
        current_task["total_reward"] = row['total_reward']
        current_task["elapsed_time"] = row["elapsed_time"]

# 添加最后一个任务
if current_task["total_reward"] != 0:
    current_task["end_index"] = df.shape[0] - 1
    tasks.append(current_task)

# 打印每个任务的总奖励和起始/结束索引
for task in tasks:
    print(f"Total Reward: {task['total_reward']}, Total Rows: {task['total_row']}, Elapsed Time: {task['elapsed_time']}")

# 打印任务的数量
print(f"Total Number of Tasks: {len(tasks)}")
# 创建输出结果的数据框
result_df = pd.DataFrame({"Task": [i + 1 for i in range(len(tasks))], "Total Rows": [task['total_row'] for task in tasks],
                         "Total Reward": [task['total_reward'] for task in tasks], "Elapsed Time": [task['elapsed_time'] for task in tasks]})

# 将结果保存为CSV文件
result_df.to_csv('out.csv', index=False)