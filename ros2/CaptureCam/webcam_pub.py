import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class ImagePublisher(Node):
    def __init__(self, name):
        # 继承父类Node的对象
        super().__init__(name)
        # 创建发布者
        self.publisher_ = self.create_publisher(Image,'image_raw',10)
        # 创建定时器
        self.timer = self.create_timer(0.1,self.time_callback)
        # 创建视频采集对象
        self.cap = cv2.VideoCapture("http://192.168.1.102:5000/video_feed")
        # 转换对象
        self.cv_bridge = CvBridge()

    def time_callback(self):
        # 读取图像
        ret,frame = self.cap.read()
        
        # 如果读取成功则发布消息
        if ret == True:
            self.publisher_.publish(
                self.cv_bridge.cv2_to_imgmsg(frame,'bgr8')
            )
        self.get_logger().info('Publishing vedio frame')

def main(args=None):
    rclpy.init(args=args)
    node = ImagePublisher('webcam_pub')
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
