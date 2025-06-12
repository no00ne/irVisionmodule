#!/usr/bin/env python3
import rospy, actionlib
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from geometry_msgs.msg import Twist
from std_msgs.msg import String
from dynamic_reconfigure.client import Client as DRClient

class SafetySupervisor:
    def __init__(self):
        rospy.init_node('traffic_supervisor')
        # 与 move_base 建立 action 客户端
        self.mb = actionlib.SimpleActionClient('move_base', MoveBaseAction)
        self.mb.wait_for_server()
        # 保存被打断的目标
        self.last_goal = None
        # cmd_vel 安全通道
        self.cmd_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        # 可选：局部规划器动态参数
        self.dwa_client = DRClient('/move_base/DWAPlannerROS', timeout=5.0)

        rospy.Subscriber('/move_base/goal', MoveBaseAction._type.goal, self._goal_cb)
        rospy.Subscriber('/traffic_action', String, self._action_cb)

    # 记录导航目标
    def _goal_cb(self, goal):
        self.last_goal = goal

    def _action_cb(self, msg: String):
        act = msg.data.lower()
        if act == 'stop':
            self._do_stop()
        elif act == 'slow':
            self._do_slow(vel_scale=0.2)
        else:                       # go
            self._do_resume()

    # 停车：取消目标 + 发零速
    def _do_stop(self):
        self.mb.cancel_all_goals()
        self._send_zero()
        # (可选)把 DWA 最大速度调到 0，防止 planner 继续发速度
        self.dwa_client.update_configuration({'max_vel_x': 0.0,
                                              'max_vel_trans': 0.0})

    # 减速
    def _do_slow(self, vel_scale=0.2):
        # 将 DWAPlanner 最大速度调低
        self.dwa_client.update_configuration({'max_vel_x': 0.3 * vel_scale,
                                              'max_vel_trans': 0.3 * vel_scale})

    # 恢复导航
    def _do_resume(self):
        # 恢复参数
        self.dwa_client.update_configuration({'max_vel_x': 0.3,
                                              'max_vel_trans': 0.3})
        # 若目标被取消且还在 last_goal，重新发
        if self.last_goal and not self.mb.get_state() in [1, 0]:  # 1=PENDING,0=ACTIVE
            self.mb.send_goal(self.last_goal)
        # 同时允许 planner 输出原速, 此处不用额外 Twist

    def _send_zero(self):
        t = Twist()
        self.cmd_pub.publish(t)

if __name__ == '__main__':
    SafetySupervisor()
    rospy.spin()
