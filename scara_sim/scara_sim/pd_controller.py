import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
import numpy as np
import os
from ament_index_python.packages import get_package_share_directory

class PDControllerNode(Node):
    def __init__(self):
        super().__init__('pd_voltage_controller_node')

        # ------------------ Параметры регулятора ------------------
        self.declare_parameter('kp1', 9.0)  
        self.declare_parameter('kd1', 5.0)   
        self.declare_parameter('kp2', 6.0)  
        self.declare_parameter('kd2', 1.0)   
        self.declare_parameter('kp3', 4.0)   # <-- НОВОЕ: 3-й контур
        self.declare_parameter('kd3', 1.0)   # <-- НОВОЕ
        
        self.kp1 = self.get_parameter('kp1').value
        self.kd1 = self.get_parameter('kd1').value
        self.kp2 = self.get_parameter('kp2').value
        self.kd2 = self.get_parameter('kd2').value
        self.kp3 = self.get_parameter('kp3').value
        self.kd3 = self.get_parameter('kd3').value

        self.declare_parameter('voltage_scale', 1.0)
        self.voltage_scale = self.get_parameter('voltage_scale').value

        self.declare_parameter('trajectory_file', 'trajectory_angles.csv')
        
        # ------------------ НАСТРОЙКА ТОПИКОВ ------------------
        self.declare_parameter('joint_states_topic', '/joint_states')
        self.declare_parameter('voltage_cmd_topic', '/voltage_commands')

        # ------------------ Загрузка траектории ------------------
        pkg_dir = get_package_share_directory('scara_sim')
        csv_path = os.path.join(pkg_dir, 'data', self.get_parameter('trajectory_file').value)
        
        try:
            data = np.loadtxt(csv_path, delimiter=',', skiprows=1)
            self.time_array = data[:, 0]
            self.q1_ref_array = data[:, 1]
            self.q2_ref_array = data[:, 2]
            self.q3_ref_array = data[:, 3]   # <-- НОВОЕ: 3-й столбец
            self.get_logger().info(f'Траектория загружена. Точек: {len(self.time_array)}')
        except Exception as e:
            self.get_logger().error(f'Ошибка загрузки траектории: {e}')
            raise

        # ------------------ Состояние робота ------------------
        self.current_q1 = 0.0
        self.current_q2 = 0.0
        self.current_q3 = 0.0                 # <-- НОВОЕ
        self.current_q1_dot = 0.0
        self.current_q2_dot = 0.0
        self.current_q3_dot = 0.0             # <-- НОВОЕ
        self.joint_state_received = False

        # ------------------ ROS 2 Интерфейсы ------------------
        self.joint_state_sub = self.create_subscription(
            JointState,
            self.get_parameter('joint_states_topic').value,
            self.joint_state_callback,
            10
        )

        self.voltage_pub = self.create_publisher(
            Float64MultiArray,
            self.get_parameter('voltage_cmd_topic').value,
            10
        )

        # Частота цикла управления (1000 Гц)
        self.control_dt = 0.001
        self.timer = self.create_timer(self.control_dt, self.control_loop)

        now_msg = self.get_clock().now().to_msg()
        self.start_time = now_msg.sec + now_msg.nanosec * 1e-9
        
        self.get_logger().info(f'ПД-регулятор напряжения (3 звена) запущен. Топик: {self.get_parameter("voltage_cmd_topic").value}')

    def joint_state_callback(self, msg: JointState):
        try:
            self.current_q1 = msg.position[0]
            self.current_q2 = msg.position[1]
            self.current_q3 = msg.position[2]   # <-- НОВОЕ
            
            if len(msg.velocity) >= 3:
                self.current_q1_dot = msg.velocity[0]
                self.current_q2_dot = msg.velocity[1]
                self.current_q3_dot = msg.velocity[2]   # <-- НОВОЕ
                
            self.joint_state_received = True
        except IndexError:
            self.get_logger().warn('Неверный формат JointState: ожидалось минимум 3 сустава.')

    def control_loop(self):
        if not self.joint_state_received:
            return

        now_msg = self.get_clock().now().to_msg()
        now = now_msg.sec + now_msg.nanosec * 1e-9
        current_time = now - self.start_time

        if current_time > self.time_array[-1]:
            current_time = self.time_array[-1]

        # Интерполяция эталонных углов (для всех 3 суставов)
        q1_ref = np.interp(current_time, self.time_array, self.q1_ref_array)
        q2_ref = np.interp(current_time, self.time_array, self.q2_ref_array)
        q3_ref = np.interp(current_time, self.time_array, self.q3_ref_array)   

        dt_small = 0.0001
        t_prev = max(0, current_time - dt_small)
        t_next = min(self.time_array[-1], current_time + dt_small)
        
        q1_prev = np.interp(t_prev, self.time_array, self.q1_ref_array)
        q2_prev = np.interp(t_prev, self.time_array, self.q2_ref_array)
        q3_prev = np.interp(t_prev, self.time_array, self.q3_ref_array)   
        
        q1_next = np.interp(t_next, self.time_array, self.q1_ref_array)
        q2_next = np.interp(t_next, self.time_array, self.q2_ref_array)
        q3_next = np.interp(t_next, self.time_array, self.q3_ref_array)   
        
        # Центральные разности для точности
        q1_dot_ref = (q1_next - q1_prev) / (2 * dt_small)
        q2_dot_ref = (q2_next - q2_prev) / (2 * dt_small)
        q3_dot_ref = (q3_next - q3_prev) / (2 * dt_small)   

        # Ошибки по положению и скорости
        error_q1 = q1_ref - self.current_q1
        error_q2 = q2_ref - self.current_q2
        error_q3 = q3_ref - self.current_q3                   
        
        error_q1_dot = q1_dot_ref - self.current_q1_dot
        error_q2_dot = q2_dot_ref - self.current_q2_dot
        error_q3_dot = q3_dot_ref - self.current_q3_dot      
        effort1 = self.kp1 * error_q1 + self.kd1 * error_q1_dot
        effort2 = self.kp2 * error_q2 + self.kd2 * error_q2_dot
        effort3 = self.kp3 * error_q3 + self.kd3 * error_q3_dot   

        # Преобразование усилия в напряжение
        voltage1 = effort1 * self.voltage_scale
        voltage2 = effort2 * self.voltage_scale
        voltage3 = effort3 * self.voltage_scale              


        # Публикация
        cmd_msg = Float64MultiArray()
        cmd_msg.data = [float(voltage1), float(voltage2), float(voltage3)]  
        self.voltage_pub.publish(cmd_msg)

def main(args=None):
    rclpy.init(args=args)
    node = PDControllerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        zero_cmd = Float64MultiArray()
        zero_cmd.data = [0.0, 0.0, 0.0]   # <-- ИЗМЕНЕНО: 3 нуля
        node.voltage_pub.publish(zero_cmd)
        node.get_logger().info('Отправлено 0В. Узел остановлен.')
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()