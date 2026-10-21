from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        
        # 1. Ваша существующая нода модели двигателя (effort_publisher)
        Node(
            package='scara_sim',
            executable='effort_publisher',
            name='effort_publisher',
            output='screen'
        ),

        # 2. Добавляем запуск ПД-регулятора
        Node(
            package='scara_sim',
            executable='pd_controller',
            name='pd_controller',
            output='screen',
          
        ),
    ])