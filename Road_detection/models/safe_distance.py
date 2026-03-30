# -*- coding: utf-8 -*-
class DrivingSafetySystem:
    # 路面摩擦系数表（需根据实际数据调整）
    ROAD_FRICTION = {
        '干燥': 0.7,  # 干燥沥青路面
        '湿滑': 0.4,  # 湿滑路面
        '结冰': 0.1,  # 结冰路面
        '积雪': 0.2,  # 积雪路面
        '吹雪': 0.15,  # 吹雪路面
        '融雪': 0.25  # 融雪路面
    }

    def __init__(self, reaction_time=1.5, g=9.81):
        """
        :param reaction_time: 驾驶员反应时间（秒）
        :param g: 重力加速度（m/s²）
        """
        self.reaction_time = reaction_time
        self.g = g

    def calculate_safe_distance(self, speed, road_condition):
        """
        计算安全制动距离
        :param speed: 当前车速（m/s）
        :param road_condition: 路面状态
        :return: 安全距离（米）
        """
        mu = self.ROAD_FRICTION.get(road_condition, 0.7)  # 默认使用干燥路面系数
        reaction_distance = speed * self.reaction_time  # 反应距离
        braking_distance = (speed ** 2) / (2 * mu * self.g)  # 制动距离
        return reaction_distance + braking_distance

    def get_warning_level(self, actual_distance, speed, road_condition):
        """
        获取预警等级
        :param actual_distance: 实际车距（米）
        :param speed: 当前车速（m/s）
        :param road_condition: 路面状态
        :return: 预警等级字符串
        """
        safe_distance = self.calculate_safe_distance(speed, road_condition)

        # 安全余量阈值设定（可根据需求调整）
        safety_margin = safe_distance * 1.2  # 建议保持的安全距离
        danger_threshold = safe_distance * 0.8  # 危险阈值

        if actual_distance >= safety_margin:
            return "安全驾驶"
        elif actual_distance >= danger_threshold:
            return "一级预警（保持警惕）"
        else:
            return "二级预警（立即制动！）"


# 使用示例
if __name__ == "__main__":
    safety_system = DrivingSafetySystem()

    # 输入参数（示例值）
    test_conditions = [
        {'speed': 45, 'distance': 150, 'road': '干燥'},  # 安全
        {'speed': 20, 'distance': 250, 'road': '结冰'},  # 危险
        {'speed': 20, 'distance': 500, 'road': '湿滑'}  # 警告
    ]

    for condition in test_conditions:
        warning = safety_system.get_warning_level(
            actual_distance=condition['distance'],
            speed=condition['speed'],
            road_condition=condition['road']
        )
        print(f"车速: {condition['speed']}m/s, 实际车距: {condition['distance']}m, 路况: {condition['road']}")
        print(f"预警状态: {warning}\n")