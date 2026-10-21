import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point

class TrajectoryVisualizer(Node):
    def __init__(self):
        super().__init__('trajectory_visualizer')
        self.publisher = self.create_publisher(Marker, '/trajectory_marker', 10)
        self.declare_parameter('publish_frequency', 10.0)
        self.publish_frequency = self.get_parameter('publish_frequency').value
        self.timer = self.create_timer(1/self.publish_frequency, self.publish_marker)
        self.publish_marker()

    def publish_marker(self):
        marker = Marker()
        marker.header.frame_id = 'world'
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'reference_trajectory'
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = 0.01
        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 1.0

        # УМЕНЬШЕННАЯ ЗВЕЗДА (идентична trajectory_generator.py)
        points = [
            ( 0.0000,  0.3000, 0.15),  # 0: верх
            ( 0.0980,  0.1348, 0.15),  # 1: впадина
            ( 0.2853,  0.0929, 0.15),  # 2: право-верх
            ( 0.1589, -0.0458, 0.15),  # 3: впадина
            ( 0.1767, -0.2425, 0.15),  # 4: право-низ
            ( 0.0000, -0.1700, 0.15),  # 5: низ (впадина)
            (-0.1767, -0.2425, 0.15),  # 6: лево-низ
            (-0.1589, -0.0458, 0.15),  # 7: впадина
            (-0.2853,  0.0929, 0.15),  # 8: лево-верх
            (-0.0980,  0.1348, 0.15),  # 9: впадина
            ( 0.0000,  0.3000, 0.15),  # замыкаем контур
        ]

        for x, y, z in points:
            p = Point()
            p.x = x
            p.y = y
            p.z = z
            marker.points.append(p)

        self.publisher.publish(marker)

def main(args=None):
    rclpy.init(args=args)
    node = TrajectoryVisualizer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()