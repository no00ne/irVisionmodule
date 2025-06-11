#!/usr/bin/env python3
import rospy
import cv2
import argparse
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from irVisionmodule.ZebraCrossing.detect import detect_zebra_frame  # 导入斑马线检测函数
from irVisionmodule.traffic_light.detect import detect_image as detect_traffic_light  # 导入交通信号灯检测函数
from irVisionmodule.ZebraCrossing.models.common import DetectMultiBackend  # 导入模型加载类
from irVisionmodule.ZebraCrossing.utils.torch_utils import select_device
from irVisionmodule.ZebraCrossing.utils.datasets import LoadStreams, LoadImages


class TrafficDetectionNode:
    def __init__(self):
        # 初始化 ROS 节点
        rospy.init_node('traffic_detection_node', anonymous=True)

        # 初始化 CvBridge
        self.bridge = CvBridge()

        # 初始化图像订阅者
        self.image_sub = rospy.Subscriber('/camera/image_raw', Image, self.image_callback)

        # 初始化交通信号发布者
        self.traffic_pub = rospy.Publisher('/traffic_action', String, queue_size=10)

        # 设置模型参数
        self.opt = self.get_opt()
        self.device = select_device(self.opt.device)
        self.model = DetectMultiBackend(self.opt.weights, device=self.device)
        self.model.eval()

        # 加载数据流
        is_stream = self.opt.source.isnumeric() or self.opt.source.endswith('.txt')
        if is_stream:
            self.dataset = LoadStreams(self.opt.source, img_size=(640, 640), stride=self.model.stride, auto=self.model.pt)
        else:
            self.dataset = LoadImages(self.opt.source, img_size=(640, 640), stride=self.model.stride, auto=self.model.pt)

    def get_opt(self):
        # 设置参数
        parser = argparse.ArgumentParser()
        parser.add_argument('--weights', type=str, default="irVisionmodule\\ZebraCrossing\\runs\\train\\exp2\\weights\\best.pt")
        parser.add_argument('--source', type=str, default='0', help='camera index or file path')
        parser.add_argument('--conf-thres', type=float, default=0.25)
        parser.add_argument('--iou-thres', type=float, default=0.45)
        parser.add_argument('--device', default='')
        parser.add_argument('--view-img', action='store_true')
        opt = parser.parse_args()
        return opt

    def image_callback(self, msg):
        # 将 ROS 图像消息转换为 OpenCV 图像
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            rospy.logerr("CvBridge Error: {0}".format(e))
            return

        # 从相机数据获取每一帧图像
        for path, img, im0s, vid_cap, s in self.dataset:
            im0 = im0s.copy() if not isinstance(im0s, list) else im0s[0]

            # 调用斑马线检测
            im0_zebra, suggestion_zebra, _, _ = detect_zebra_frame(
                im0, self.model, self.device, self.opt.conf_thres, self.opt.iou_thres, self.opt.view_img
            )

            # 调用交通信号灯检测
            im0_traffic, suggestion_traffic = detect_traffic_light(im0)

            # 显示图像
            cv2.imshow("Zebra Crossing Detection", im0_zebra)  # 斑马线检测窗口
            cv2.imshow("Traffic Light Detection", im0_traffic)  # 交通信号灯检测窗口

            # 打印交通信号建议
            rospy.loginfo(f"Zebra Crossing Suggestion: {suggestion_zebra}")
            rospy.loginfo(f"Traffic Light Suggestion: {suggestion_traffic}")

            # 发布交通信号动作（可以根据需要进行处理，或使用 ROS 发布）
            action = "GO"
            if suggestion_zebra == 'stop' or suggestion_traffic == 'stop':
                action = "STOP"
            elif suggestion_zebra == 'slow' or suggestion_traffic == 'slow':
                action = "SLOW"
            self.traffic_pub.publish(action)

            # 按 "q" 键退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        node = TrafficDetectionNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
