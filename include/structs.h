#pragma once
#include <iostream>
#include <vector>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <string>
#include <utility>

struct LidarSLAMObject {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    int flag1 = -1;// whether to add a pose prior
  int flag2 = -1;// whether to add a pose change prior

  int object_id = -1;
  int frame_id = -1;
  int obj_status = -1;
  int type = -1;
  std::vector<float> local_xyz{ 0,0,0 };
  std::vector<float> rotation_zyx{ 0,0,0 };
  std::vector<float> measure_lwh{ 0,0,0 };
  float pose_inLidar[6] = { 0,0,0,0,0,0 }; //roll pitch yaw x y z
  bool dynamic = false;
  bool initialized = false;
  float score = 0;
  std::vector<float> optimize_t{ 0,0,0,0,0,0 };//roll pitch yaw x y z
  std::vector<float> v_t{ 0,0,0,0,0,0 };
  double velocity = 0;
  int vertex_id = -1;

  int edge_id_link_frame_pose = -1;
  bool associated = false;
  std::pair<int, int> associated_object_last = std::make_pair(-1, -1);
  std::pair<int, int> associated_object_next = std::make_pair(-1, -1);
  int vertex_id_pose_change = -1;
  int edge_id_link_2object_with_pose_change = -1;
  bool continuous_associated = false; // Whether continuously associated (three frames and more)
  int edge_id_link_2pose_change = -1;

};

struct LidarSLAMFrame {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    int frame_id = -1;
  Eigen::Affine3f optimize_pose = Eigen::Affine3f::Identity();
  int vertex_local_id = -1;
  int vertex_global_id = -1;
  int edge_local_id = -1;
  std::vector<LidarSLAMObject> objects;
};

struct LidarPoint {
  float x;
  float y;
  float z;
  float intensity;
  int32_t time_offset;
  LidarPoint() {
    x = 0;
    y = 0;
    z = 0;
    intensity = 0;
    time_offset = 0;
  }
  LidarPoint(float _x, float _y, float _z, float _intensity) {
    x = _x;
    y = _y;
    z = _z;
    intensity = _intensity;
    time_offset = 0;
  }
  LidarPoint(float _x, float _y, float _z, float _intensity,
    int32_t _time_offset) {
    x = _x;
    y = _y;
    z = _z;
    intensity = _intensity;
    time_offset = _time_offset;
  }
};

struct OneFrame {
  int64_t timestamp;
  std::vector<LidarPoint> frame_data;
};

