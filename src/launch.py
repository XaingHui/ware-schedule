from src.env.envrioment import WarehouseEnvironment, Item
import csv
from datetime import datetime


def add_items_from_csv(env, csv_file):
    with open(csv_file, 'r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)  # 跳过 CSV 文件的标题行

        for row in csv_reader:
            item_id = row[0]
            length = int(row[1])
            width = int(row[2])
            start_time = str(row[3])
            processing_time = int(row[4])
            exit_time = str(row[5])

            # 计算物品的 x 和 y 坐标，根据统一行宽的前一个物品去计算
            # x, y = env.calculate_item_position(length, width)

            # 添加物品到环境
            env.add_item(item_id, length, width, start_time, processing_time, exit_time)


def launch():
    # 创建环境实例
    env = WarehouseEnvironment(width=105, height=85, number=50)

    # 示例用法：从CSV文件中读取货物信息并添加到环境
    add_items_from_csv(env, 'data1.csv')

    # # 进行其他初始化步骤，如创建代理、设置超参数等
    # agent = YourAgent()
    # agent.set_hyperparameters()
    #
    # # 开始训练或测试循环
    # for episode in range(num_episodes):
    #     state = env.reset()
    #     done = False
    #     total_reward = 0
    #
    #     while not done:
    #         # 代理选择动作并执行
    #         action = agent.choose_action(state)
    #         new_state, reward, done, _ = env.step(action)
    #         total_reward += reward
    #
    #         # 更新代理的学习过程
    #
    #         state = new_state
    #
    #     # 打印每个周期的总奖励
    #     print(f"Episode {episode}: Total Reward = {total_reward}")
    #
    # # 最后，你可以保存训练好的模型等等
    # agent.save_model('trained_agent')

    # 显示环境
    env.render()


if __name__ == "__main__":
    launch()
