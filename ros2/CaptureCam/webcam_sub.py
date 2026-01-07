import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from ultralytics import YOLO
import os

class ImageSubscriber(Node):
    def __init__(self, name):
        super().__init__(name)
        self.sub = self.create_subscription(Image,'image_raw',self.listener_callback,10)
        self.cv_bridge = CvBridge()
        self.model = YOLO(os.path.expanduser('~/model/yolov8n.pt'))
        
    def obj_detect(self,image):
        img_rgb = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        
        result = self.model.predict(img_rgb, verbose=False, imgsz=640, device=0)
            
        annotated_img = result[0].plot()
        annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR)

        cv2.imshow('Yolov8 Detection',annotated_img)
        cv2.waitKey(1)    
        
    def listener_callback(self,data):
        self.get_logger().info('Listening vedio fram')
        image = self.cv_bridge.imgmsg_to_cv2(data,'bgr8')
        self.obj_detect(image)
        
def main(args=None):
    rclpy.init(args=args)
    node = ImageSubscriber('webcam_sub')
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
