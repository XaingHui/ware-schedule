import time

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
    def get_rectangle(self):
        """
        获取物品的矩形形状。

        返回：
        - (left, top, right, bottom) 元组表示矩形的左上角和右下角坐标
        """
        left = self.x
        top = self.y
        right = self.x + self.width
        bottom = self.y + self.length
        return left, top, right, bottom

    def move(self, x, y):
        self.x = x
        self.y = y


class WarehouseEnvironment:
    def __init__(self, width, height, number, time='2017/9/1'):
        self.width = width
        self.height = height
        self.number = number
        self.segment_heights = [20, 19, 18, 16, 15, 13, 13, 11, 10, 9, 8]  # 存储已添加物品的分段高度
        self.grid = np.zeros((height, width + 20), dtype=object)
        self.agent_position = (0, 0)
        self.items = {}
        self.colors = list(mcolors.TABLEAU_COLORS)
        self.road = {'x': width, 'width': 20, 'color': 'lightgray'}  # 道路属性
        self.target_position = (0, 0)  # 目标位置列表
        self.current_time = datetime.strptime(time, '%Y/%m/%d')  # 最早出场时间
        self.initial_state = {
            'agent_position': self.agent_position,
            'target_positions': self.target_position,
        }
        self.cache_items = []
        self.start_time = datetime.now()
        self.get_target_position()

    def get_target_position(self, x=0, y=0):
        self.target_position = (x, y)

    def simulate_time_passage(self):
        # 判断是否过了24秒，如果是，增加一天
        start_time = self.start_time.second
        end_time = datetime.now().second
        hours = abs(int(end_time - start_time))
        self.current_time += timedelta(hours=hours)

    def get_state(self):
        agent_position = self.agent_position  # 代理机器人的位置
        target_positions = self.target_position  # 目标位置

        state = {
            'agent_position': agent_position,
            'target_positions': target_positions
        }

        return state

    def step(self, action):
        move_x_distance = 1  # 默认移动距离
        move_y_distance = 1  # 默认移动距离
        distance_x_to_target = abs(self.agent_position[0] - self.target_position[0])
        distance_y_to_target = abs(self.agent_position[1] - self.target_position[1])
        if distance_x_to_target > 50:
            move_x_distance = int(distance_x_to_target / 2)  # 如果距离大于20，则调整移动距离
        elif distance_x_to_target > 30 < 50:
            move_x_distance = int(distance_x_to_target / 2)    # 如果距离大于20，则调整移动距离
        elif distance_x_to_target > 20 < 30:
            move_x_distance = int(distance_x_to_target / 2)
        elif distance_x_to_target > 10 < 20:
            move_x_distance = int(distance_x_to_target / 2)
        # 如果距离大于20，则调整移动距离
        if distance_y_to_target > 50:
            move_y_distance = int(distance_y_to_target / 2)
            # 如果距离大于20，则调整移动距离
        elif distance_y_to_target > 30 < 50:
            move_y_distance = int(distance_y_to_target / 2)
            # 如果距离大于20，则调整移动距离
        elif distance_y_to_target > 20 < 30:
            move_y_distance = int(distance_y_to_target / 2) # 如果距离大于20，则调整移动距离
        elif distance_y_to_target > 10 < 20:
            move_y_distance = int(distance_y_to_target / 2)
        reward = 0
        item = Item('B000', 0, 0, 0, 0, '2017/9/1', 0, '2017/9/29', 'red')
        if self.target_position == (0, 0):
            # 随机获取一个物品的坐标
            item = np.random.choice(list(self.items.values()))
            # 获取目标位置
            self.get_target_position(item.x, item.y)

        # 记录之前的代理机器人位置
        prev_agent_position = self.agent_position
        if self.agent_position[0] > self.target_position[0]:
            reward -= 100
        if self.agent_position[1] > self.target_position[1]:
            reward -= 100

        # 执行动作并更新环境状态
        if action == 0:  # 代理机器人向上移动
            self.agent_position = (self.agent_position[0], max(0, self.agent_position[1] - move_y_distance))
            time.sleep(0.01 * move_y_distance)  # 模拟移动的时间
        elif action == 1:  # 代理机器人向下移动
            self.agent_position = (self.agent_position[0], min(self.height, self.agent_position[1] + move_y_distance))
            time.sleep(0.01 * move_y_distance)
        elif action == 2:  # 代理机器人向左移动
            self.agent_position = (max(0, self.agent_position[0] - move_x_distance), self.agent_position[1])
            time.sleep(0.01 * move_x_distance)
        elif action == 3:  # 代理机器人向右移动
            self.agent_position = (min(self.width , self.agent_position[0] + move_x_distance), self.agent_position[1])
            time.sleep(0.01 * move_x_distance)
        else:
            print("Invalid action!")
            reward = -100

        # 计算奖励
        x_distance_to_target = abs(self.agent_position[0] - self.target_position[0])
        y_distance_to_target = abs(self.agent_position[1] - self.target_position[1])
        reward += 300.0 - x_distance_to_target - y_distance_to_target  # 根据距离计算奖励

        # 检查是否到达目标位置，根据情况返回奖励
        if self.agent_position == self.target_position:
            reward += 300  # 到达目标位置的奖励
            if self.target_position[0] >= 75:
                self.target_position = (0, 0)
                # 根据目标坐标找到item

                done = True  # 任务完成
            else:
                self.target_position = self.width, item.y
                done = False
        else:
            done = False

        self.simulate_time_passage()
        # 更新状态
        new_state = self.get_state()
        self.render()  # 更新环境

        return new_state, reward, done, {}

    def reset(self):
        # Reset the environment to its initial state
        self.agent_position = (0, 0)
        self.items = {}
        self.grid = np.zeros((self.height, self.width), dtype=int)
        state = self.initial_state  # Reset the state to the initial state
        return state

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

    def remove_item(self, item):
        """
        从环境中删除指定的物品对象。

        参数：
        - item: 要删除的物品对象

        返回：
        - 无返回值
        """
        # 从环境中删除物品
        del self.items[(item.x, item.y)]

        # 从 grid 中清除物品标识
        self.grid[item.y, item.x] = None
        self.render()

    def check_item(self, item_id, x, y, length, width, start_time, processing_time, exit_time):
        item = self.add_item(item_id, x, y, length, width, start_time, processing_time, exit_time)
        if self.current_time >= item.start_time:
            size = item.length * item.width
            self.items[(item.x, item.y)] = item
            # 将物品信息添加到环境网格中
            for i in range(size):
                for j in range(size):
                    self.grid[item.y, item.x] = len(self.items)

    def check_collision(self, item1, item2):
        """
        检查两个矩形是否相交。

        参数：
        - rectangle1: 第一个矩形的坐标 (left, top, right, bottom)
        - rectangle2: 第二个矩形的坐标 (left, top, right, bottom)

        返回：
        - 如果矩形相交，则返回 True，否则返回 False
        """
        rectangle1 = item1.get_rectangle()
        rectangle2 = item2.get_rectangle()
        if not (rectangle1[2] < rectangle2[0] or  # 左
                rectangle1[0] > rectangle2[2] or  # 右
                rectangle1[3] < rectangle2[1] or  # 上
                rectangle1[1] > rectangle2[3]):
            return True

        return item2

    def move_to_target_row(self, item, target_row):
        # 检查是否满足条件搬出目标方块
        if item.length <= target_row and abs(item.length - target_row) <= 2:
            # 可以搬出目标方块
            tmp_item = item
            self.remove_item(item)
            start_time = str(item.start_time).replace('-', '/').strip(' 00:00:00')
            exit_time = str(item.exit_time).replace('-', '/').strip(' 00:00:00')

            self.check_item(tmp_item.item_id, tmp_item.x, tmp_item.y - target_row, tmp_item.length,
                            tmp_item.width, start_time,
                            tmp_item.processing_time, exit_time)
        self.render()

    def move_to_target_position(self, item, target_position):
        """
        将物品移动到目标位置

        参数：
        - item: 要移动的物品对象
        - target_position: 目标位置

        返回：
        - 无返回值
        """
        # 检查目标位置是否合法
        if target_position < 0 or target_position >= \
                self.width + self.road['width']:
            raise ValueError("目标位置不合法")
        # item_target = Item('tmp',)
        # if self.check_collision() and self.move_out_to_other_row(item, target_position):
        #     self.move_interference_item(interference_item, target_position)

        tmp_item = item
        self.remove_item(item)
        start_time = str(item.start_time).replace('-', '/').strip(' 00:00:00')
        exit_time = str(item.exit_time).replace('-', '/').strip(' 00:00:00')

        self.check_item(tmp_item.item_id, tmp_item.x, target_position, tmp_item.length,
                        tmp_item.width, start_time,
                        tmp_item.processing_time, exit_time)

    def move_interference_item(self, item_to_move, target_row):
        """
        递归地处理干涉方块，将它们移动到目标行。

        参数：
        - item_to_move: 要移动的物品对象
        - target_row: 目标行

        返回：
        - 无返回值
        """
        # 移动物品并检查冲突
        for new_y in range(target_row, target_row + item_to_move.y):
            item_to_move.move(new_y)
            # 在这里检查是否与其他物品冲突
            for other_item in self.items.values():
                if other_item != item_to_move:
                    if self.check_collision(item_to_move, other_item):
                        # 处理冲突，可能需要采取适当的措施
                        print(f"冲突发生：{item_to_move.item_id} 与 {other_item.item_id}")
                        # 递归地处理干涉方块
                        self.move_interference_item(other_item, target_row)

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
    env.check_item('B003', 8, 114, 11, 8, '2017/9/1', 13, '2017/9/22')
    env.check_item('B002', 0, 101, 13, 11, '2017/9/1', 16, '2017/9/29')
    env.move_to_target_position(env.get_item_by_id('B001'), 74)
    env.render()
    env.move_to_target_row(env.get_item_by_id('B003'), 11)


if __name__ == "__main__":
    main()
