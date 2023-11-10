import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from datetime import datetime, timedelta


class Item:
    def __init__(self, item_id, x, y, length, width, start_time, processing_time, exit_time, color):
        self.item_id = item_id
        self.length = length  # 分段长
        self.width = width  # 分段宽
        self.start_time = datetime.strptime(start_time, '%Y/%m/%d')  # 最早开工时间
        self.processing_time = processing_time  # 加工周期
        self.exit_time = datetime.strptime(exit_time, '%Y/%m/%d')  # 最早出场时间
        self.x = x  # 物品的 x 坐标
        self.y = y  # 物品的 y 坐标
        self.color = color  # 可选：为物品添加颜色属性

    def __str__(self):
        return f"Item ID: {self.item_id}, Size: ({self.length}m x {self.width}m), Bound:({self.x + self.width},{self.y + self.length})" \
               f"Start Time: {self.start_time}, Processing Time: {self.processing_time} days, " \
               f"Exit Time: {self.exit_time}"

    # 可以根据需要添加其他方法，例如移动物品或检查物品与其他物品的冲突

    def move(self, x, y):
        self.x = x
        self.y = y


class WarehouseEnvironment:
    def __init__(self, width, height, number, time='2017/9/1', time_speed=24):
        self.time_counter = 0
        self.width = width
        self.height = height
        self.number = number
        self.segment_heights = [20, 19, 18, 16, 15, 13, 13, 11, 10, 9, 8]  # 存储已添加物品的分段高度
        self.grid = np.zeros((height, width), dtype=int)
        self.agent_position = (0, 0)
        self.interference_blocks = []
        self.items = {}
        self.colors = list(mcolors.TABLEAU_COLORS)
        self.road = {'x': width, 'width': 20, 'color': 'lightgray'}  # 道路属性
        self.target_positions = []  # 目标位置列表
        self.current_time = datetime.strptime(time, '%Y/%m/%d')  # 最早出场时间
        self.initial_state = self.get_state()
        self.cache_items = []
        self.start_time = datetime.now()
        self.get_target_positions()
        self.get_interference_blocks()

    def get_target_positions(self):
        for i in range(self.height):
            self.target_positions.append((self.width, i))

    def get_interference_blocks(self):
        pass

    def simulate_time_passage(self):
        # 判断是否过了24秒，如果是，增加一天
        start_time = self.start_time.second
        end_time = datetime.now().second
        hours = abs(int(end_time - start_time))
        self.current_time += timedelta(hours=hours)

    def get_state(self):
        agent_position = self.agent_position  # 代理机器人的位置
        target_positions = [list(pos) for pos in self.target_positions]  # 目标位置列表
        item_positions = [list(pos) for pos in self.items.keys()]  # 物品位置列表
        interference_positions = [list(pos) for pos in self.interference_blocks]  # 干涉物位置列表
        # 将所有部分合并为一个 NumPy 数组
        state = {
            'agent_position': agent_position,
            'target_positions': target_positions,
            'item_positions': item_positions,
            'interference_positions': interference_positions
        }

        return state

    def step(self, action):

        # 执行动作并更新环境状态
        if action == -1:  # 代理机器人向上移动
            self.agent_position = (self.agent_position[0], max(0, self.agent_position[1] - 1))
        elif action == 1:  # 代理机器人向下移动
            self.agent_position = (self.agent_position[0], min(self.height - 1, self.agent_position[1] + 1))
        else:
            raise ValueError("Invalid action")

        # 检查是否到达目标位置，根据情况返回奖励
        if self.agent_position in self.target_positions:
            reward = 300  # 到达目标位置的奖励
            done = True  # 任务完成
        else:
            reward = 0.0  # 没有到达目标位置的奖励
            done = False

        self.simulate_time_passage()
        # 更新状态
        new_state = self.get_state()

        return new_state, reward, done, {}

    def reset(self):
        # Reset the environment to its initial state
        self.agent_position = (0, 0)
        self.items = {}
        self.interference_blocks = []
        self.grid = np.zeros((self.height, self.width), dtype=int)
        self.state = self.initial_state  # Reset the state to the initial state
        return self.state

    """
    添加物品到 环境当中
    """

    def add_item(self, item_id, x, y, length, width, start_time, processing_time, exit_time):
        if len(self.items) >= self.number:
            # 如果空地上的物品数量已经达到限制，你可以采取适当的操作，例如引发异常
            raise ValueError("空地上的物品数量已经达到限制")
        item_color = self.colors[len(self.items) % len(self.colors)]
        item = Item(item_id, x, y, length, width, start_time, processing_time, exit_time, item_color)
        return item

    def check_item(self, item_id, x, y, length, width, start_time, processing_time, exit_time):
        item = self.add_item(item_id, x, y, length, width, start_time, processing_time, exit_time)
        if self.current_time >= item.start_time:
            size = item.length * item.width
            self.items[(item.x, item.y)] = item
            # 将物品信息添加到环境网格中
            for i in range(size):
                for j in range(size):
                    self.grid[item.y, item.x] = len(self.items)

    def move_out_to_other_row(self, item, target_row):
        # 检查是否满足条件搬出目标方块
        if abs(item.length - target_row) <= 2:
            # 可以搬出目标方块
            return True
        return False

    def handle_interference(self, interference_items, method=1):
        if method == 1:
            # 方法1：先将干涉方块搬出场地，目标方块搬出后，将干涉方块按出场时间降序排列，重新放入场内
            interference_items.sort(key=lambda item: item['exit_time'], reverse=True)
            for item in interference_items:
                self.add_item(item['x'], item['y'], item['size'])

        elif method == 2:
            # 方法2：将干涉方块移至相邻的上下行中，待目标方块搬出后不放回原所在行
            for item in interference_items:
                new_y = item['y'] + 1 if item['y'] > self.agent_position[1] else item['y'] - 1
                if 0 <= new_y < self.height:
                    self.add_item(item['x'], new_y, item['size'])

        elif method == 3:
            # 方法3：如果上下行没有方块阻挡，可直接从邻行搬出
            for item in interference_items:
                upper_y = item['y'] - 1
                lower_y = item['y'] + item['size']
                if (0 <= upper_y) and (lower_y < self.height):
                    if not any(self.grid[upper_y:lower_y, item['x']]):
                        self.add_item(item['x'], upper_y, item['size'])

    def move_interference_items(self, item_to_move, target_row):
        """
        递归地处理干涉方块，将它们移动到目标行。

        参数：
        - item_to_move: 要移动的物品对象
        - target_row: 目标行

        返回：
        - 无返回值
        """
        # 检查干涉方块
        interference_items = self.find_interference_items(item_to_move.x, target_row)

        if interference_items:
            for interference_item in interference_items:
                # 移动干涉方块到目标行
                new_y = target_row + 1 if interference_item.y < target_row else target_row - 1
                self.move_item(interference_item, new_y)

                # 递归调用，处理可能继续的干涉方块
                self.move_interference_items(interference_item, target_row)

    def move_item(self, item, target_position):
        """
        将物品移动到目标位置

        参数：
        - item: 要移动的物品对象
        - target_position: 目标位置

        返回：
        - 无返回值
        """
        # 检查目标位置是否合法
        if target_position < 0 or target_position >= self.width:
            raise ValueError("目标位置不合法")

        # 检查目标位置是否有物品
        if self.grid[item.y, target_position] != 0:
            raise ValueError("目标位置已经有物品")

        # 检查目标位置是否有干涉方块
        interference_items = self.find_interference_items(target_position, item.y)
        if interference_items:
            raise ValueError("目标位置有干涉方块")

        # 移动物品
        self.grid[item.y, item.x] = 0
        self.grid[item.y, target_position] = item.id
        item.x = target_position

    def render(self):
        plt.figure(figsize=(5, 5))
        # 创建 x 轴刻度标签
        x_ticks = [0, self.road['x'], self.width]

        # 绘制刻度标签
        plt.xticks(x_ticks, fontsize=8)
        plt.imshow(np.ones((self.height, self.width + 20)), cmap='binary', interpolation='none', origin='upper')

        # 创建一个列表来存储y轴刻度的位置
        current_y = 0
        y_positions = []
        for i in range(len(self.segment_heights)):
            y_positions.append(current_y)
            current_y += self.segment_heights[i]

        # 设置y轴刻度标签的位置和标签

        plt.yticks(y_positions, self.segment_heights, fontsize=8)
        for (x, y), item in self.items.items():
            rect = plt.Rectangle((x, y), item.width, item.length, color=item.color, alpha=0.5)
            plt.gca().add_patch(rect)
            # 添加文本 "box1" 到方块内部
            text_x = x + item.width / 2
            text_y = y + item.length / 2
            plt.text(text_x, text_y, item.item_id, ha='center', va='center', fontsize=6, color='black')

        road_x = self.road['x']
        road_width = self.road['width']
        road_color = self.road['color']
        road_rect = plt.Rectangle((road_x, 0), road_width, self.height, color=road_color, alpha=1)
        plt.gca().add_patch(road_rect)
        plt.show()

    def get_item_by_id(self, id):
        for item in self.items.values():
            if item.item_id == id:
                return item


def main():
    # 创建环境实例
    env = WarehouseEnvironment(width=75, height=153, number=50)

    # 示例用法：添加物品并显示环境
    env.check_item('B001', 0, 114, 11, 8, '2017/9/1', 13, '2017/9/22')
    env.check_item('B002', 0, 101, 13, 11, '2017/9/1', 16, '2017/9/29')
    env.render()


if __name__ == "__main__":
    main()
