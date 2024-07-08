# LIMOT
<!-- * The code will be released after the paper is accepted but we provide a [gtsam demo](https://github.com/Sonne-Zhu/gtsam_demo_limot) to illustrate and verify that gtsam is used correctly in LIMOT. -->
* A tightly-coupled multi-object tracking and LiDAR-inertial odometry system, allowing for joint estimation of the poses of both the ego-vehicle and surrounding objects.
<p align='center'>
    <img src="./doc/demo.gif" alt="drawing" width="800"/>
</p>

## Dependency

We developed LIMOT on Ubuntu 20.04.
- [ROS](http://wiki.ros.org/ROS/Installation) (Noetic)
  ```
  sudo apt-get install -y ros-noetic-navigation
  sudo apt-get install -y ros-noetic-robot-localization
  sudo apt-get install -y ros-noetic-robot-state-publisher
  ```
- [gtsam](https://gtsam.org/get_started/) (Georgia Tech Smoothing and Mapping library, recommended 4.0.3)
  ```
  sudo add-apt-repository ppa:borglab/gtsam-release-4.0
  sudo apt install libgtsam-dev libgtsam-unstable-dev
  ```
## Install

Use the following commands to download and compile the package.

```
mkdir -p ~/limot_ws/src
cd ~/limot_ws/src
git clone https://github.com/tiev-tongji/LIMOT.git
cd ..
catkin_make -j
```

## Run the package

1. Run the launch file:
```
source devel/setup.bash
roslaunch limot run_kitti.launch
```
2. Run your object detector, which subscribes to LiDAR scans and publishes the detection results with the formats: 

    [timestamp, [type, x, y, z, l, w, h, yaw, score], ...,[type ,x, ..., score]].

3. Play existing bag files:
```
rosbag play your-bag.bag
```
## Sample dataset

* Download the KITTI tracking dataset to test the functionality of the LIMOT. The dataset below are configured to run using the [params_kitti.yaml](./config/params_kitti.yaml):
    - **KITTI tracking dataset:** [[Google Drive](https://drive.google.com/drive/folders/144Kp2WYfHIF6SbKHDCtjTswOk2E1_gPg?usp=sharing)]

* Download the self-collected dataset to test the functionality of the LIMOT. The dataset below are configured to run using the [params_hdl64.yaml](./config/params_hdl64.yaml):
    - **Self-collected dataset:** [[Google Drive](https://drive.google.com/drive/folders/1-30POMAEe8F7kGfUjNHDNh8gOYRHjOt2?usp=sharing)]   
    

## Paper

* LIMOT has been accepted by IEEE Robotics and Automation Letters.
```
@article{zhu2024limot,
  title={LIMOT: a tightly-coupled system for LiDAR-inertial odometry and multi-object tracking},
  author={Zhu, Zhongyang and Zhao, Junqiao and Huang, Kai and Tian, Xuebo and Lin, Jiaye and Ye, Chen},
  journal={IEEE Robotics and Automation Letters},
  year={2024},
  publisher={IEEE}
}
```

* Our previous work DL-SLOT has been accepted by the IEEE Transactions on Intelligent Vehicles.
```
@article{tian2023dl,
  title={DL-SLOT: Tightly-Coupled Dynamic LiDAR SLAM and 3D Object Tracking Based on Collaborative Graph Optimization},
  author={Tian, Xuebo and Zhu, Zhongyang and Zhao, Junqiao and Tian, Gengxuan and Ye, Chen},
  journal={IEEE Transactions on Intelligent Vehicles},
  year={2023},
  publisher={IEEE}
}
```

## Acknowledgments
Thanks for LOAM(J. Zhang and S. Singh. LOAM: Lidar Odometry and Mapping in Real-time) and [LIO-SAM](https://github.com/TixiaoShan/LIO-SAM).