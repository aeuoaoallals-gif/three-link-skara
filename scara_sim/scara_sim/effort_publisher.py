import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
import numpy as np

class MotorModelNode(Node):
    def __init__(self):
        super().__init__('motor_model_node')

        self.K_m = 0.05
        self.R = 2.0
        self.L = 0.01
        self.K_e = 0.05
        self.J_motor = 0.001
        self.J_reducer = 0.001
        self.i_gear = 10.0
        self.tau_e = self.L / self.R
        self.dt = 0.001

        self.L1 = 0.37
        self.L2 = 0.25
        self.L3 = 0.20
        self.rho = 2800.0
        self.d = 0.05

        self.m1 = self.rho * np.pi * (self.d/2)**2 * self.L1
        self.m2 = self.rho * np.pi * (self.d/2)**2 * self.L2
        self.m3 = self.rho * np.pi * (self.d/2)**2 * self.L3

        self.l1 = self.L1 / 2
        self.l2 = self.L2 / 2
        self.l3 = self.L3 / 2

        self.I1zz = (1/12) * self.m1 * (3*(self.d/2)**2 + self.L1**2)
        self.I2zz = (1/12) * self.m2 * (3*(self.d/2)**2 + self.L2**2)
        self.I3zz = (1/12) * self.m3 * (3*(self.d/2)**2 + self.L3**2)

        self.b1 = 0.1
        self.b2 = 0.1
        self.b3 = 0.1

        self.M_dv1 = 0.0
        self.M_dv2 = 0.0
        self.M_dv3 = 0.0
        self.omega1 = 0.0
        self.omega2 = 0.0
        self.omega3 = 0.0
        self.U1 = 0.0
        self.U2 = 0.0
        self.U3 = 0.0
        self.q1 = 0.0
        self.q2 = 0.0
        self.q3 = 0.0
        self.q1_dot = 0.0
        self.q2_dot = 0.0
        self.q3_dot = 0.0
        self.q1_ddot = 0.0
        self.q2_ddot = 0.0
        self.q3_ddot = 0.0
        self.q1_dot_prev = 0.0
        self.q2_dot_prev = 0.0
        self.q3_dot_prev = 0.0
        self.joint_state_received = False

        self.joint_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_state_callback, 10)
        self.voltage_sub = self.create_subscription(
            Float64MultiArray, '/voltage_commands', self.voltage_callback, 10)
        self.effort_pub = self.create_publisher(
            Float64MultiArray, '/effort_controller/commands', 10)
        self.timer = self.create_timer(self.dt, self.control_loop)

    def joint_state_callback(self, msg: JointState):
        if len(msg.position) >= 3 and len(msg.velocity) >= 3:
            self.q1 = msg.position[0]
            self.q2 = msg.position[1]
            self.q3 = msg.position[2]
            self.q1_dot = msg.velocity[0]
            self.q2_dot = msg.velocity[1]
            self.q3_dot = msg.velocity[2]

            self.q1_ddot = (self.q1_dot - self.q1_dot_prev) / self.dt
            self.q2_ddot = (self.q2_dot - self.q2_dot_prev) / self.dt
            self.q3_ddot = (self.q3_dot - self.q3_dot_prev) / self.dt

            self.q1_dot_prev = self.q1_dot
            self.q2_dot_prev = self.q2_dot
            self.q3_dot_prev = self.q3_dot
            self.joint_state_received = True

    def voltage_callback(self, msg: Float64MultiArray):
        if len(msg.data) >= 3:
            self.U1 = msg.data[0]
            self.U2 = msg.data[1]
            self.U3 = msg.data[2]

    def compute_load_torque_and_inertia(self, q1, q2, q3, q1_dot, q2_dot, q3_dot, q1_ddot, q2_ddot, q3_ddot):
        c2 = np.cos(q2)
        c3 = np.cos(q3)
        c23 = np.cos(q2 + q3)
        s2 = np.sin(q2)
        s3 = np.sin(q3)
        s23 = np.sin(q2 + q3)

        M11 = (self.I1zz + self.I2zz + self.I3zz +
               self.m1*self.l1**2 +
               self.m2*(self.L1**2 + self.l2**2 + 2*self.L1*self.l2*c2) +
               self.m3*(self.L1**2 + self.L2**2 + self.l3**2 +
                        2*self.L1*self.L2*c2 + 2*self.L1*self.l3*c23 + 2*self.L2*self.l3*c3))

        M12 = (self.I2zz + self.I3zz +
               self.m2*(self.l2**2 + self.L1*self.l2*c2) +
               self.m3*(self.L2**2 + self.l3**2 + self.L1*self.L2*c2 +
                        self.L1*self.l3*c23 + 2*self.L2*self.l3*c3))

        M13 = self.I3zz + self.m3*(self.l3**2 + self.L1*self.l3*c23 + self.L2*self.l3*c3)

        M22 = (self.I2zz + self.I3zz + self.m2*self.l2**2 +
               self.m3*(self.L2**2 + self.l3**2 + 2*self.L2*self.l3*c3))

        M23 = self.I3zz + self.m3*(self.l3**2 + self.L2*self.l3*c3)

        M33 = self.I3zz + self.m3*self.l3**2

        h2 = (self.m2*self.L1*self.l2*s2 +
              self.m3*self.L1*self.L2*s2 +
              self.m3*self.L1*self.l3*s23)
        h3 = self.m3*self.L2*self.l3*s3

        C11 = -h2*q2_dot - h3*(q2_dot + q3_dot)
        C12 = -h2*(q1_dot + q2_dot) - h3*(q1_dot + q2_dot + q3_dot)
        C13 = -h3*(q1_dot + q2_dot + q3_dot)
        C21 =  h2*q1_dot - h3*q3_dot
        C22 = -h3*q3_dot
        C23 = -h3*(q1_dot + q2_dot + q3_dot)
        C31 =  h3*(q1_dot + q2_dot)
        C32 =  h3*(q1_dot + q2_dot)
        C33 =  0.0

        tau1 = (M11*q1_ddot + M12*q2_ddot + M13*q3_ddot +
                C11*q1_dot + C12*q2_dot + C13*q3_dot + self.b1*q1_dot)
        tau2 = (M12*q1_ddot + M22*q2_ddot + M23*q3_ddot +
                C21*q1_dot + C22*q2_dot + C23*q3_dot + self.b2*q2_dot)
        tau3 = (M13*q1_ddot + M23*q2_ddot + M33*q3_ddot +
                C31*q1_dot + C32*q2_dot + C33*q3_dot + self.b3*q3_dot)

        J_links1 = M11 / (self.i_gear**2)
        J_links2 = M22 / (self.i_gear**2)
        J_links3 = M33 / (self.i_gear**2)

        return tau1, tau2, tau3, J_links1, J_links2, J_links3

    def control_loop(self):
        if not self.joint_state_received:
            return

        tau1, tau2, tau3, J_l1, J_l2, J_l3 = self.compute_load_torque_and_inertia(
            self.q1, self.q2, self.q3,
            self.q1_dot, self.q2_dot, self.q3_dot,
            self.q1_ddot, self.q2_ddot, self.q3_ddot)

        M_n_m1 = tau1 / self.i_gear
        M_n_m2 = tau2 / self.i_gear
        M_n_m3 = tau3 / self.i_gear

        J_t1 = max(self.J_motor + self.J_reducer + J_l1, 0.001)
        J_t2 = max(self.J_motor + self.J_reducer + J_l2, 0.001)
        J_t3 = max(self.J_motor + self.J_reducer + J_l3, 0.001)

        om1 = self.q1_dot * self.i_gear
        om2 = self.q2_dot * self.i_gear
        om3 = self.q3_dot * self.i_gear

        U_eff1 = self.U1 - self.K_e * om1
        self.M_dv1 += (((self.K_m / self.R) * U_eff1 - self.M_dv1) / self.tau_e) * self.dt
        self.omega1 += ((self.M_dv1 - M_n_m1) / J_t1) * self.dt

        U_eff2 = self.U2 - self.K_e * om2
        self.M_dv2 += (((self.K_m / self.R) * U_eff2 - self.M_dv2) / self.tau_e) * self.dt
        self.omega2 += ((self.M_dv2 - M_n_m2) / J_t2) * self.dt

        U_eff3 = self.U3 - self.K_e * om3
        self.M_dv3 += (((self.K_m / self.R) * U_eff3 - self.M_dv3) / self.tau_e) * self.dt
        self.omega3 += ((self.M_dv3 - M_n_m3) / J_t3) * self.dt

        cmd = Float64MultiArray()
        cmd.data = [
            float(self.M_dv1 * self.i_gear),
            float(self.M_dv2 * self.i_gear),
            float(self.M_dv3 * self.i_gear)
        ]
        self.effort_pub.publish(cmd)

def main(args=None):
    rclpy.init(args=args)
    node = MotorModelNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        zero_cmd = Float64MultiArray()
        zero_cmd.data = [0.0, 0.0, 0.0]
        node.effort_pub.publish(zero_cmd)
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()