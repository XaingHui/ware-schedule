from collections import defaultdict, deque

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from datetime import datetime


class Item:
    def __init__(self, item_id, length, width, start_time, processing_time, exit_time, color):
        self.item_id = item_id
        self.length = length  # 分段长
        self.width = width  # 分段宽
        self.start_time = datetime.strptime(start_time, '%Y/%m/%d')  # 最早开工时间
        self.processing_time = processing_time  # 加工周期
        self.exit_time = datetime.strptime(exit_time, '%Y/%m/%d')  # 最早出场时间
        self.x = 0  # 物品的 x 坐标
        self.y = 0  # 物品的 y 坐标
        self.color = color  # 可选：为物品添加颜色属性

    def __str__(self):
        return f"Item ID: {self.item_id}, Size: ({self.length}m x {self.width}m), " \
               f"Start Time: {self.start_time}, Processing Time: {self.processing_time} days, " \
               f"Exit Time: {self.exit_time}"

    # 可以根据需要添加其他方法，例如移动物品或检查物品与其他物品的冲突


class WarehouseEnvironment:
    def __init__(self, width, height, number):
        self.width = width
        self.height = height
        self.number = number
        self.segment_width_queues = {}  # 存储不同分段宽度的队列
        self.segment_widths = []  # 存储已添加物品的分段宽度
        self.segment_heights = []  # 存储已添加物品的分段高度
        self.grid = np.zeros((height, width), dtype=int)
        self.agent_position = (0, 0)
        self.target_positions = [(width - 1, i) for i in range(height)]
        self.interference_blocks = []
        self.items = {}
        self.colors = list(mcolors.TABLEAU_COLORS)
        self.road = {'x': width, 'width': 20, 'color': 'lightgray'}  # 道路属性
        # Initialize the initial state
        self.item_cache = []  # 创建一个缓存队列来存储待添加的物品
        self.queue_by_row_width = defaultdict(deque)

        self.initial_state = self.get_state()


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

    def add_item(self, item_id, length, width, start_time, processing_time, exit_time):
        if len(self.items) >= self.number:
            # 如果空地上的物品数量已经达到限制，你可以采取适当的操作，例如引发异常
            raise ValueError("空地上的物品数量已经达到限制")
            # 如果空地上的物品数量已经达到限制，将物品添加到缓存队列等待
            # self.item_cache.append((item_id, length, width, start_time, processing_time, exit_time, item_color))
        if len(self.items) == 0:
            x = 0
            y = 0
        if len(self.items) == 1:
            x = 0
            first_item = list(self.items.values())[-1]
            y = self.height - first_item.length
        if len(self.items) > 1:
            # 过滤物品 根据不同的height
            dict_items = self.filter_items()
            for index in self.segment_heights:
                list_items = list(dict_items[f'items_{index}'])
                prev_item = list_items[-1]
                prev_x = prev_item.x
                prev_width = prev_item.width
                current_item = list_items[0]
                if current_item.length != prev_item.length:
                    # 计算当前物品的坐标
                    x = 0
                    y = self.height - prev_item.length
                else:
                    x = prev_x + prev_width
                    y = self.height - prev_item.length

        # 存储分段宽度
        # self.segment_widths.append(width)
        if sum(self.segment_heights) < self.height:
            self.segment_heights.append(length)
        else:
            pass
            # 这里写如何处理多出来的物品，比如添加到队列中保存

        if (x, y) not in self.items:
            item_color = self.colors[len(self.items) % len(self.colors)]
            item = Item(item_id, length, width, start_time, processing_time, exit_time, item_color)
            size = length * width
            item.x = x
            item.y = y
            item.size = length * width
            # item.color = color
            self.items[(x, y)] = item
            # self.grid[y, x] = item.x, item.y
            # 将物品添加到对应行宽的队列
            self.queue_by_row_width[width].append(item)
            # 将物品信息添加到环境网格中
            for i in range(size):
                for j in range(size):
                    self.grid[y, x] = len(self.items)

            return item

    def move_out_target_item(self, x, y, target_row):
        # 检查是否满足条件搬出目标方块
        if abs(y - target_row) <= 2:
            # 可以搬出目标方块
            return True
        return False

    def filter_item(self, height):
        filter_list = list(filter(lambda item: item[1].length == height, self.items.items()))
        result = []
        for _, item in filter_list:
            result.append(item)
        return result

    def filter_items(self):
        dict_items = {}
        dict_item = {}
        for height in self.segment_heights:
            height_index = 'height_' + str(height)
            dict_item[height_index] = height
            items_index = 'items_' + str(height)
            dict_item[items_index] = self.filter_item(height)
            # print(dict_item)
            dict_items.update(dict_item)
        return dict_items

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

    def move_item(self, x, y, size, target_row):
        if self.move_out_target_item(x, y, target_row):
            # 可以搬出目标方块
            self.add_item(x, target_row, size)
        else:
            # 满足条件的搬运方法
            interference_items = []
            for item_x in range(x, x + size):
                interference_items.extend(self.find_interference_items(item_x, target_row))

            if len(interference_items) > 0:
                # 有干涉方块，选择合适的方法
                self.handle_interference(interference_items, method)

    def calculate_y_position(self):
        current_y = 0  # 初始化y坐标
        for segment_width in self.unique_segment_widths:
            self.y_positions.append(self.height - (current_y + segment_width))
            current_y += segment_width

    def render(self):

        plt.figure(figsize=(5, 5))
        # 创建 x 轴刻度标签
        x_ticks = [0, self.road['x'], self.width]

        # 绘制刻度标签
        plt.xticks(x_ticks, fontsize=8)
        plt.imshow(np.ones((self.height, self.width + 20)), cmap='binary', interpolation='none', origin='upper')
        # unique_segment_widths = list(set(self.segment_widths))
        unique_segment_heights = list(set(self.segment_heights))

        # 创建一个列表来存储y轴刻度的位置
        y_positions = []
        current_y = 0  # 初始化y坐标
        # for segment_width in unique_segment_widths:
        #     y_positions.append(self.height - (current_y + segment_width))
        #     current_y += segment_width
        for segment_height in unique_segment_heights:
            y_positions.append(self.height - (current_y + segment_height))
            current_y += segment_height

        # 设置y轴刻度标签的位置和标签
        plt.yticks(y_positions, unique_segment_heights, fontsize=8)

        for row_width, item_queue in self.queue_by_row_width.items():
            for item in item_queue:
                x, y = item.x, item.y
                rect = plt.Rectangle((x, y), item.width, item.length, color=item.color, alpha=0.5)
                plt.gca().add_patch(rect)

        road_x = self.road['x']
        road_width = self.road['width']
        road_color = self.road['color']
        road_rect = plt.Rectangle((road_x, 0), road_width, self.height, color=road_color, alpha=1)
        plt.gca().add_patch(road_rect)
        plt.show()


def main():
    # 创建环境实例
    env = WarehouseEnvironment(width=75, height=105, number=50)

    # 示例用法：添加物品并显示环境
    item1 = env.add_item('B001', 11, 8, '2017/9/1', 13, '2017/9/22')
    item2 = env.add_item('B002', 13, 11, '2017/9/5', 16, '2017/9/29')
    env.render()


if __name__ == "__main__":
    main()
