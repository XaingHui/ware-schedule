from src.env.envrioment import WarehouseEnvironment
import csv
from datetime import datetime


def read_items_from_csv(csv_file):
    items = []  # 存储物品信息的列表

    with open(csv_file, 'r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)  # 跳过CSV文件的标题行

        for row in csv_reader:
            item = {
                'ID': row[0],
                '分段长/m': int(row[1]),
                '分段宽/m': int(row[2]),
                '最早开工时间': datetime.strptime(row[3], '%Y/%m/%d'),
                '加工周期/de': int(row[4]),
                '最早出场时间': datetime.strptime(row[5], '%Y/%m/%d')
            }
            items.append(item)

    return items


def launch():
    # 创建环境实例
    env = WarehouseEnvironment(width=150, height=75)

    # 示例用法：从CSV文件中读取货物信息并添加到环境
    items = read_items_from_csv('data1.csv')
    for item in items:
        env.add_item(item['x'], item['y'], item['size'])

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

    # 假设按照CSV文件中的顺序添加物品
    agent_x, agent_y = 0, 0  # 代理机器人的初始位置
    for item in items:
        item_x = agent_x + 1  # 假设将物品放在代理机器人的右侧
        item_y = agent_y
        item_size = item['分段长/m']

        # 将物品添加到环境
        env.add_item(item_x, item_y, item_size)

        # 更新代理机器人的位置，以便下一个物品放在它的右侧
        agent_x = item_x + item_size

    # 显示环境
    env.render()


if __name__ == "__main__":
    launch()
