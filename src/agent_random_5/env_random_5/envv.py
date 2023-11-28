import csv
import operator
import time
from random import choice

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from datetime import datetime, timedelta


class Item:
    def __init__(self, item_id, x, y, length, width, start_time, processing_time, exit_time, color):
        self.item_id = item_id
        self.length = length  # 分段长
        self.width = width  # 分段宽
        start_time = str(start_time).replace('-', '/').strip(' 00:00:00')
        exit_time = str(exit_time).replace('-', '/').strip(' 00:00:00')
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


class WarehouseEnvironment:
    def __init__(self, width, height, number, time='2017/9/1'):
        self.prev_target_position = (0, 0)
        self.width = width
        self.height = height
        self.number = number
        self.segment_heights = [20, 20, 20, 20, 20, 20]  # 存储已添加物品的分段高度
        self.grid = np.zeros((height, width + 20), dtype=object)
        self.agent = Item('agent', 0, 0, 5, 5, time, 0, time, 'black')
        self.agent_position = self.agent.x, self.agent.y
        self.items = {}
        self.colors = list(mcolors.TABLEAU_COLORS)
        self.road = {'x': width, 'width': 20, 'color': 'lightgray'}  # 道路属性
        self.target_position = (0, 0)  # 目标位置
        self.current_time = datetime.strptime(time, '%Y/%m/%d')  # 最早出场时间
        self.agent_has_item = False
        self.total_reward = 0
        self.total_step_time = 0
        self.item = Item('tmp', self.agent.x, self.agent.y, 5, 5, '2017/9/1', 0, '2017/9/1', 'black')
        self.task_positions = []
        self.item_random = None
        self.conflict_count = 0
        self.initial_state = {
            'agent_has_item': self.agent_has_item,
            'agent_position': self.agent_position,
            'target_positions': self.target_position,
        }
        self.cache_items = []
        self.step_records = []
        self.interfering_items = []
        self.start_time = datetime.now()

    def simulate_time_passage(self):
        # 判断是否过了1秒，如果是，增加一分钟
        start_time = self.start_time.second
        end_time = datetime.now().second
        hours = abs(int(end_time - start_time))
        self.current_time += timedelta(minutes=hours * 3)

    def get_state(self):
        agent_position = self.agent_position  # 代理机器人的位置
        target_positions = self.target_position  # 目标位置

        state = {
            'agent_position': agent_position,
            'target_positions': target_positions
        }

        return state

    def binary_forward(self):
        move_x_distance = 1  # 默认移动距离
        move_y_distance = 1  # 默认移动距离
        distance_x_to_target = abs(self.agent.x - self.target_position[0])
        distance_y_to_target = abs(self.agent.y - self.target_position[1])

        if distance_x_to_target > self.width / 2:
            move_x_distance = int(distance_x_to_target / 2)  # 如果距离大于20，则调整移动距离
        elif distance_x_to_target > self.width / 4 < self.width / 2:
            move_x_distance = int(distance_x_to_target / 2)
        elif distance_x_to_target > self.width / 8 < self.width / 4:
            move_x_distance = int(distance_x_to_target / 2)
        elif distance_x_to_target > self.width / 16 < self.width / 8:
            move_x_distance = int(distance_x_to_target / 2)

        if distance_y_to_target > self.height / 2:
            move_y_distance = int(distance_y_to_target / 2)
        elif distance_y_to_target > self.height / 4 < self.height / 2:
            move_y_distance = int(distance_y_to_target / 2)
        elif distance_y_to_target > self.height / 8 < self.height / 4:
            move_y_distance = int(distance_y_to_target / 2)
        elif distance_y_to_target > self.height / 16 < self.height / 8:
            move_y_distance = int(distance_y_to_target / 2)

        return move_x_distance, move_y_distance

    def has_cache_item(self):
        if len(self.cache_items) > 0:
            for i in range(len(self.cache_items)):
                item = self.cache_items.pop(i)
                start_time = str(item.start_time).replace('-', '/').strip(' 00:00:00')
                exit_time = str(item.exit_time).replace('-', '/').strip(' 00:00:00')
                if self.check_item(item.item_id, item.x, item.y, item.length, item.width, start_time,
                                   item.processing_time, exit_time):
                    break
                else:
                    break

    def record_step(self, action, reward, done):
        step_info = {
            'action': action,
            'agent_position': self.agent_position,
            'target_position': self.target_position,
            'total_reward': self.total_reward,
            'elapsed_time': self.total_step_time,
            'conflict_count': self.conflict_count,
        }
        self.step_records.append(step_info)
        if done:
            self.save_records_to_csv()
            self.total_step_time = 0
            self.total_reward = 0
            self.conflict_count = 0

    def agent_move(self, action, move_x_distance, move_y_distance):
        # 执行动作并更新环境状态
        if action == 0:  # 代理机器人向上移动
            self.agent_position = (self.agent.x, max(0, self.agent.y - move_y_distance))
            self.agent.x, self.agent.y = self.agent_position
            time.sleep(0.001 * move_y_distance)  # 模拟移动的时间
        elif action == 1:  # 代理机器人向下移动
            self.agent_position = (self.agent.x, min(self.height, self.agent.y + move_y_distance))
            if self.agent_position[1] + self.agent.length > self.height:
                self.agent_position = (self.agent.x, self.height - self.agent.length)
            else:
                self.agent.x, self.agent.y = self.agent_position
            time.sleep(0.001 * move_y_distance)
        elif action == 2:  # 代理机器人向左移动
            self.agent_position = (max(0, self.agent.x - move_x_distance), self.agent.y)
            self.agent.x, self.agent.y = self.agent_position
            time.sleep(0.001 * move_x_distance)
        elif action == 3:  # 代理机器人向右移动
            self.agent_position = (min(self.width, self.agent.x + move_x_distance), self.agent.y)
            self.agent.x, self.agent.y = self.agent_position
            time.sleep(0.001 * move_x_distance)
        else:
            print("Invalid action!")
            reward = -100


    def get_earliest_item(self):
        if len(self.items) > 0 and self.target_position == (0, 0):
            # 获取最早出场时间的物品
            earliest_item = min(self.items.values(), key=operator.attrgetter('exit_time'))
            if earliest_item.exit_time <= self.current_time:
                # 设置抽取的物品
                self.item_random = earliest_item
                self.item = self.item_random

                # 获取目标位置
                self.task_positions.append((self.item.x, self.item.y))
                self.target_position = self.task_positions.pop(-1)
            else:
                pass

    def step(self, action):
        print("---------------------------------------------------------------")
        print(self.current_time)

        print("现在的物品有:")
        for k, v in self.items.items():
            print(k, v.item_id)
        for interfering in self.interfering_items:
            print("干涉物品是：  " + str(interfering.item_id) + "  " + str(interfering.x) + "  " + str(interfering.y))
        if self.agent_has_item:
            print("Agent has item!       " + str(self.agent.item_id))
        # 记录每一步的时间
        done = False
        step_time = datetime.now()
        #  检测是否有缓存的物品需要加入
        self.has_cache_item()
        # 快速移动
        move_x_distance, move_y_distance = self.binary_forward()
        # 奖励初始化
        reward = 0

        if self.target_position == (0, 0):
            if len(self.items) == 0 and len(self.cache_items) == 0:
                if len(self.task_positions) == 0:
                    done = True
                    reward = 10000
                    new_state = self.get_state()
                    self.total_step_time = 0
                    self.total_step_time = round(self.total_step_time, 5)
                    # 记录每一步的信息
                    self.record_step(action, reward, done)
                    return new_state, reward, done, {}
                done = False
                self.target_position = self.task_positions.pop(0)

        # 获取最早出场时间的物品
        self.get_earliest_item()
        # 执行动作并更新环境状态
        self.agent_move(action, move_x_distance, move_y_distance)

        # 计算奖励
        if self.target_position != (0, 0) and self.target_position[0] < 75:
            x_distance_to_target = abs(self.agent.x - self.target_position[0])
            y_distance_to_target = abs(self.agent.y - self.target_position[1])
            reward += 300.0 - x_distance_to_target - y_distance_to_target  # 根据距离计算奖励

        if self.agent.x > self.target_position[0]:
            reward -= 100
        if self.agent.y > self.target_position[1]:
            reward -= 100

        if self.agent_position == self.target_position:
            print("现在干涉物品的长度：  " + str(len(self.interfering_items)))
            if len(self.interfering_items) != 0 and self.agent_has_item is False and self.target_position[0] < 75:
                item = self.interfering_items[-1]
                print("干扰物品的是：", item.item_id)
                print("干扰物品的位置：", item.x, item.y)
                print("机器人到达要添加物品的位置：", item.x, item.y)
                item = self.interfering_items.pop(-1)
                start_time = str(item.start_time).replace('-', '/').strip(' 00:00:00')
                exit_time = str(item.exit_time).replace('-', '/').strip(' 00:00:00')
                self.check_item(item.item_id, item.x, item.y, item.length, item.width, start_time,
                                item.processing_time,
                                exit_time)
            # 代理机器人到达目标位置
            if self.target_position[0] < 75 and self.target_position != (0, 0):
                if self.item_random is None:
                    item = self.items.get((self.target_position[0], self.target_position[1]))
                    print("目标物品的是：", item.item_id)
                    self.item = item
                    self.remove_item(item)
                    self.agent = self.item
                    self.agent_has_item = True
                else:
                    self.item = self.item_random
                    self.remove_item(self.item_random)
                    self.agent = self.item

                    self.agent_has_item = True

            if len(self.agent.item_id) < 10:
                self.agent.item_id = 'agent_' + str(self.item.item_id)
            self.agent.color = 'red'
            reward += 100  # 到达目标位置的奖励
            if self.target_position[0] >= 75:
                self.item = self.getInitItem()
                self.agent = self.item
                self.agent_has_item = False
                reward += 500  # 成功搬运物品的奖励
                self.task_positions.append((0, 0))
                self.target_position = self.task_positions.pop(-1)
                self.item_random = None
                # self.reset()
                if len(self.task_positions) == 0:
                    done = True  # 任务完成
            elif self.agent_has_item is True and len(self.task_positions) == 0:
                self.prev_target_position = self.target_position
                self.task_positions.append((self.width, self.agent.y))
                self.target_position = self.task_positions.pop(-1)
                done = False
            elif self.agent_has_item is True and len(self.task_positions) != 0:
                self.prev_target_position = self.target_position
                self.target_position = self.task_positions.pop(-1)
                self.agent_has_item = False
                self.items.update({(self.agent.x, self.agent.y): self.agent})
                self.agent = self.getInitItem()
                done = False
        else:
            done = False

        # 在代理机器人移动过程中检测冲突
        print("代理机器人的状态： ", self.agent.item_id, self.agent.x, self.agent.y)
        print("是否携带物品：", self.agent_has_item)
        if self.agent_has_item is True:
            for other_item in list(self.items.values()):
                if other_item.item_id.strip('agent_') != self.agent.item_id.strip(
                        'agent_') and self.check_collision(self.agent, other_item):
                    # 处理冲突
                    print(
                        f"冲突发生：代理机器人携带的物品与其他物品冲突  " + other_item.item_id.strip(
                            'agent_') + "     " +
                        self.agent.item_id.strip('agent_'))
                    self.conflict_count += 1
                    self.handle_conflict(other_item)
                    reward -= 3000  # 冲突的惩罚

        if len(self.task_positions) > 0:
            print("任务位置的长度是：", len(self.task_positions))
            print("当前任务位置是：", self.target_position[0], self.target_position[1])
            print("下一步任务位置是：", self.task_positions[-1][0], self.task_positions[-1][1])

        if len(self.task_positions) > 0 and self.agent.x == self.target_position[0] \
                and self.agent.y == self.target_position[1]:
            self.target_position = self.task_positions.pop(-1)

        # 更新环境
        if self.target_position == (0, 0):
            reward = 0
        self.simulate_time_passage()
        # 更新状态
        new_state = self.get_state()
        self.clean_on_road()

        self.render()  # 更新环境

        self.total_reward += reward
        self.total_step_time += (datetime.now() - step_time).total_seconds()
        self.total_step_time = round(self.total_step_time, 5)
        self.record_step(action, reward, done)
        return new_state, reward, done, {}

    def save_records_to_csv(self):
        with open('simulation_records.csv', mode='w', newline='') as file:
            fieldnames = ['action', 'agent_position', 'target_position', 'total_reward', 'elapsed_time',
                          'conflict_count']
            writer = csv.DictWriter(file, fieldnames=fieldnames)

            writer.writeheader()
            for record in self.step_records:
                writer.writerow(record)

    def handle_conflict(self, interfering_item):
        """
        处理冲突的方式1：重新放置干涉方块
        """
        self.exchange_agent_item(interfering_item)

        print("冲突解决： 现在的agent携带的物品是  " + self.agent.item_id.strip('agent_'))
        print("冲突解决： 现在的Item携带的物品是  " + self.item.item_id.strip('agent_'))
        # self.target_position = self.task_positions.pop(-1)
        # self.task_positions.append((75, interfering_item.y))
        # self.task_positions.append((interfering_item.x, interfering_item.y))
        # self.task_positions.append((75, interfering_item.y))
        print("任务位置有： ")
        print(self.task_positions)


    def exchange_agent_item(self, interfering_item):
        """
        交换agent和interfering_item
        """
        self.task_positions.append((interfering_item.x, interfering_item.y))
        self.agent.item_id = self.agent.item_id.strip('agent_')
        self.items.update({(self.agent.x, self.agent.y): self.agent})
        for item in self.interfering_items:
            if interfering_item.id != item.item_id:
                self.interfering_items.append(interfering_item)
                break
        self.remove_item(interfering_item)
        print("干扰物品位置： " + str(interfering_item.x), str(interfering_item.y))

        self.item = interfering_item
        self.agent = self.item
        self.agent_has_item = True

    def get_target_row(self, current_item):
        """
        获取目标方块上下行中的一个可用行

        Parameters:
            current_row (int): 当前行号

        Returns:
            target_row (int): 目标方块上下行中的一个可用行，如果没有可用行则返回 None
            :param current_item:
        """
        # 检查移动到上面的行是否会发生冲突
        if current_item.length not in self.segment_heights:
            length = current_item.length + 1
        else:
            length = current_item.length
        index = self.segment_heights.index(length)
        upper_row = self.segment_heights[index - 1]
        if not self.is_conflict_with_target(upper_row, current_item):
            return upper_row

        # 检查移动到下面的行是否会发生冲突
        lower_row = self.segment_heights[index + 1]
        if not self.is_conflict_with_target(lower_row, current_item):
            return lower_row

        # 如果上面和下面的行都会发生冲突，则返回 None 表示没有可用行
        return None

    def is_conflict_with_target(self, target_row, current_item):
        """
        检查移动到目标行是否会发生冲突

        Parameters:
            target_row (int): 目标行号

        Returns:
            is_conflict (bool): 是否与目标发生冲突，如果发生冲突则为 True，否则为 False
            :param current_item:
            :param target_row:
            :param item:
        """
        # 在这里添加检查冲突的逻辑，例如检查与其他物体的碰撞等
        item = Item(current_item.item_id, current_item.x, current_item.y + target_row, current_item.length,
                    current_item.width,
                    current_item.start_time, current_item.processing_time, current_item.exit_time, current_item.color)

        is_conflict = self.check_collision(current_item, item)

        return is_conflict

    def reset(self):
        # Reset the environment to its initial state
        state = self.initial_state  # Reset the state to the initial state
        self.prev_target_position = (0, 0)
        self.agent = Item('agent', 0, 0, 5, 5, '2017/9/1', 0, '2017/9/1', 'black')
        self.agent_position = self.agent.x, self.agent.y
        self.target_position = (0, 0)  # 目标位置
        self.agent_has_item = False
        self.total_reward = 0
        self.total_step_time = 0
        self.item = Item('tmp', self.agent.x, self.agent.y, 5, 5, '2017/9/1', 0, '2017/9/1', 'black')
        self.task_positions = []
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

    # 在 remove_item 方法中添加如下行
    def remove_item(self, item):
        """
        从场地中移除物品
        """
        position = (item.x, item.y)
        del self.items[position]

    def check_item(self, item_id, x, y, length, width, start_time, processing_time, exit_time):
        """
        检查到达的物品对象。
        如果物品的到达时间早于当前时间，则将物品添加到环境中。
        如果物品的到达时间晚于当前时间，则将物品添加到缓存列表中。等到时间到达时再将物品添加到环境中。

        参数：
        - item: 到达的物品对象

        返回：
        - 无返回值
        """
        item = self.add_item(item_id, x, y, length, width, start_time, processing_time, exit_time)

        if self.current_time >= item.start_time:
            # 检查相同 y 坐标的物品
            com_y_items = self.filter_item_by_y(item.y)
            com_y_items.sort(key=lambda x: x.exit_time, reverse=True)

            if com_y_items:
                # 如果存在相同 y 坐标的物品，则设置添加物品的 x 为前面物品的 x + width
                last_item = com_y_items[0]
                item.x = last_item.x + last_item.width
            else:
                # 如果不存在相同 y 坐标的物品，则设置添加物品的 x
                item.x = item.x

            size = item.length * item.width
            self.items[(item.x, item.y)] = item
            return True
        else:
            if item not in self.cache_items:
                self.cache_items.append(item)
            return False

    def filter_item_by_y(self, y):
        """
        过滤相同物品的 y 坐标。

        :return: items列表
        """
        items_com = []
        for (k, v) in self.items.items():
            if k[1] == y:
                items_com.append(v)
        return items_com

    def check_collision(self, item1, item2):
        epsilon = 0.1  # 误差
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
        if not (rectangle1[2] + epsilon < rectangle2[0] or  # 左
                rectangle1[0] + epsilon > rectangle2[2] or  # 右
                rectangle1[3] + epsilon < rectangle2[1] or  # 上
                rectangle1[1] + epsilon > rectangle2[3]):
            return True
        return False

    def clean_on_road(self):
        """
        清除道路上的物品
        :return: item_id
        """
        for (k, v) in self.items.items():
            if k[0] >= 75:
                item_id = v.item_id
                self.remove_item(v)
                return item_id

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

        # Draw agent
        agent_x = self.agent.x
        agent_y = self.agent.y
        agent_width = self.agent.width
        agent_length = self.agent.length
        agent_color = self.agent.color

        agent_rect = plt.Rectangle((agent_x, agent_y), agent_width, agent_length, color=agent_color, alpha=0.5)
        plt.gca().add_patch(agent_rect)

        # 添加文本 "box1" 到方块内部
        text_x = agent_x + agent_width / 2
        text_y = agent_y + agent_length / 2
        plt.text(text_x, text_y, self.agent.item_id, ha='center', va='center', fontsize=6, color='black')

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

    def getInitItem(self):
        init_item = Item('tmp', self.agent.x, self.agent.y, 5, 5, '2017/9/1', 0, '2017/9/1', 'black')
        return init_item


def main():
    # 创建环境实例
    env = WarehouseEnvironment(width=75, height=153, number=50)

    # 示例用法：添加物品并显示环境
    env.check_item('B001', 0, 114, 11, 8, '2017/9/1', 13, '2017/9/22')
    env.render()
    env.check_item('B003', 8, 114, 11, 8, '2017/9/2', 13, '2017/9/22')
    env.check_item('B007', 19, 114, 11, 8, '2017/9/2', 13, '2017/9/29')
    env.check_item('B009', 40, 114, 11, 8, '2017/9/2', 13, '2017/9/27')
    env.check_item('B0013', 60, 114, 11, 8, '2017/9/2', 13, '2017/9/21')
    env.render()


if __name__ == "__main__":
    main()