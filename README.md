# Manip Sort ROS2 

ROS 2 workspace для проекта сортировки объектов с использованием UR10e, MoveIt 2 и Gazebo.

## Структура

```text
manip_sort_ros2_ws/
├── README.md
├── .gitignore
└── src/
    ├── manip_sort_bringup
    ├── manip_sort_description
    ├── manip_sort_pipeline
    └── robotiq_description
```


## Требования

- Ubuntu 24.04
- ROS 2 Jazzy
- `colcon`
- `rosdep`
- MoveIt 2
- Gazebo Harmonic / `ros_gz`
- пакеты UR: `ur_description`, `ur_moveit_config`, `ur_robot_driver`, `ur_simulation_gz`

## Сборка

```bash
cd manip_sort_ros2_ws
source /opt/ros/jazzy/setup.bash
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install
source install/setup.bash
```

## Запуск

### 1. Базовая симуляция UR10e + MoveIt 2

```bash
cd manip_sort_ros2_ws
source /opt/ros/jazzy/setup.bash
source install/setup.bash
ros2 launch manip_sort_bringup ur10e_sim_moveit.launch.py
```

### 2. Демонстрация движения по пяти точкам

```bash
cd manip_sort_ros2_ws
source /opt/ros/jazzy/setup.bash
source install/setup.bash
ros2 launch manip_sort_pipeline ur10e_five_point_demo.launch.py
```

### 3. Демонстрация сцены сортировки

```bash
cd manip_sort_ros2_ws
source /opt/ros/jazzy/setup.bash
source install/setup.bash
ros2 launch manip_sort_pipeline ur10e_sort_demo.launch.py
```

### 4. Vision-based сортировка

```bash
cd manip_sort_ros2_ws
source /opt/ros/jazzy/setup.bash
source install/setup.bash
ros2 launch manip_sort_pipeline ur10e_vision_sort_demo.launch.py
```

## Полезно

- Сборка отдельных пакетов:

```bash
colcon build --symlink-install --packages-select manip_sort_bringup manip_sort_description manip_sort_pipeline robotiq_description
```

- Запуск без GUI Gazebo:

```bash
ros2 launch manip_sort_pipeline ur10e_sort_demo.launch.py gazebo_gui:=false
```

- Vision demo в dry-run режиме:

```bash
ros2 launch manip_sort_pipeline ur10e_vision_sort_demo.launch.py
```
Значение `dry_run` задаётся в [scene_sorting.yaml](/home/krimson/diploma/manip_sort_ros2_ws/src/manip_sort_pipeline/config/scene_sorting.yaml:1).

- Быстрая smoke-проверка perception/grasp логики:

```bash
source /opt/ros/jazzy/setup.bash
source install/setup.bash
ros2 run manip_sort_perception perception_smoke_check
```

- Unit-тесты perception на синтетических сценах:

```bash
source /opt/ros/jazzy/setup.bash
source install/setup.bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q src/manip_sort_perception/test
```
