#pragma once
#include <pcl/common/common.h>
#include <pcl/common/transforms.h>

#include <dirent.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <structs.h>

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <opencv2/core/core.hpp>
#include <string>

using namespace std;
Eigen::Affine3f calib_T_from_camera_to_lidar;
bool use_true_object_position;
string data_path = "/home/tiev-vsim/data_zzy/tracking_bag/hyh_detection/";


vector<string> SplitString(string line, char tag) {
  vector<string> strvec;
  string s;
  stringstream ss(line);
  while (getline(ss, s, tag)) {
    strvec.push_back(s);
  }
  return strvec;
}

bool GetAffine3FromVector(Eigen::Affine3f& Aff, vector<string> words) {
  Eigen::Matrix4f T = Eigen::Matrix4f::Identity();
  if (words.size() == 12) {
    T(0, 0) = atof(words[0].c_str());
    T(0, 1) = atof(words[1].c_str());
    T(0, 2) = atof(words[2].c_str());
    T(0, 3) = atof(words[3].c_str());
    T(1, 0) = atof(words[4].c_str());
    T(1, 1) = atof(words[5].c_str());
    T(1, 2) = atof(words[6].c_str());
    T(1, 3) = atof(words[7].c_str());
    T(2, 0) = atof(words[8].c_str());
    T(2, 1) = atof(words[9].c_str());
    T(2, 2) = atof(words[10].c_str());
    T(2, 3) = atof(words[11].c_str());
    Aff.matrix() = T;
    return true;
  }
  else if (words.size() == 9) {
    T(0, 0) = atof(words[0].c_str());
    T(0, 1) = atof(words[1].c_str());
    T(0, 2) = atof(words[2].c_str());
    T(1, 0) = atof(words[3].c_str());
    T(1, 1) = atof(words[4].c_str());
    T(1, 2) = atof(words[5].c_str());
    T(2, 0) = atof(words[6].c_str());
    T(2, 1) = atof(words[7].c_str());
    T(2, 2) = atof(words[8].c_str());
    Aff.matrix() = T;
    return true;
  }
  else {
    cout << "error in get Affine3f from a string vector" << endl;
    return false;
  }
}

vector<LidarSLAMFrame, Eigen::aligned_allocator<LidarSLAMFrame>> GetLidarSLAMFrames_KITTI(string sequence_numebr, float Scorethre) {
  vector<LidarSLAMFrame, Eigen::aligned_allocator<LidarSLAMFrame>> frames;
  string objects_path = data_path + sequence_numebr + "/objects/";
  string time_path = data_path + sequence_numebr + "/pose.txt";
  vector<string> time_strings;
  ifstream fin_time_file(time_path);
  string time_line;
  while (getline(fin_time_file, time_line)) {
    time_strings.push_back(time_line);
  }
  string calib_path = data_path + sequence_numebr + "/calib.txt";
  vector<string> calib_strings;
  vector<Eigen::Affine3f, Eigen::aligned_allocator<Eigen::Affine3f>> calib_matrix;
  cout << "reading calib file start..... " << endl;
  ifstream fin_calib_file(calib_path);
  string calib_line;
  while (getline(fin_calib_file, calib_line)) {
    calib_strings.push_back(calib_line);
  }
  cout << "reading calib file end! have " << calib_strings.size() << " items" << endl;
  fin_calib_file.close();
  cout << "calib_strings.size() = " << calib_strings.size() << endl;
  for (int i = 0; i < calib_strings.size(); i++) {
    vector<string> calib_words = SplitString(calib_strings[i], ' ');
    Eigen::Affine3f cal_t = Eigen::Affine3f::Identity();
    calib_words.erase(calib_words.begin());
    if (GetAffine3FromVector(cal_t, calib_words)) {
      calib_matrix.push_back(cal_t);
    }
  }
  Eigen::Affine3f vel_2_cam = Eigen::Affine3f::Identity();
  int frame_num = time_strings.size();
  int n_zero = 6;
  for (int i = 0; i < frame_num; i++) {
    LidarSLAMFrame frame;
    string number = to_string(i);
    string index = std::string(n_zero - number.length(), '0') + number;
    frame.frame_id = i;
    vector<LidarSLAMObject> objects;
    ifstream fin_object_file(objects_path + index + ".txt");
    string object_line;
    int plus_num = 0;
    while (getline(fin_object_file, object_line)) {
      vector<string> words = SplitString(object_line, ' ');
      LidarSLAMObject ob;
      ob.frame_id = i;
      ob.obj_status = 1;
      ob.type = atoi(words[0].c_str());
      if (ob.type != 0)
        continue;
      ob.score = atof(words[10].c_str());
      Eigen::Vector3f local_xyz = Eigen::Vector3f::Zero();
      local_xyz = vel_2_cam * Eigen::Vector3f(atof(words[1].c_str()), atof(words[2].c_str()), atof(words[3].c_str()));
      Eigen::Vector3f rotation_zyx = Eigen::Vector3f::Zero();
      rotation_zyx = Eigen::Vector3f(atof(words[4].c_str()), atof(words[5].c_str()), atof(words[6].c_str()));
      if (local_xyz[0] <= -30.0 || abs(local_xyz[1]) > 24.0 || atof(words[10].c_str()) < Scorethre) {
        continue;
      }

      Eigen::Vector3f measure_lwh = Eigen::Vector3f::Zero();
      measure_lwh = Eigen::Vector3f(atof(words[7].c_str()), atof(words[8].c_str()), atof(words[9].c_str()));
      for (int i = 0; i < 3; i++) {
        ob.local_xyz[i] = local_xyz[i];
        ob.rotation_zyx[i] = rotation_zyx(i);
        ob.measure_lwh[i] = measure_lwh(i);
      }
      Eigen::Affine3f local_t = Eigen::Affine3f::Identity();
      local_t = pcl::getTransformation(local_xyz[0], local_xyz[1], local_xyz[2], 0, 0, rotation_zyx[0]);
      float roll, pitch, yaw, x, y, z;
      ob.pose_inLidar[0] = local_t.rotation().eulerAngles(0, 1, 2)[0];
      ob.pose_inLidar[1] = local_t.rotation().eulerAngles(0, 1, 2)[1];
      ob.pose_inLidar[2] = local_t.rotation().eulerAngles(0, 1, 2)[2];
      ob.pose_inLidar[3] = local_t.translation()[0];
      ob.pose_inLidar[4] = local_t.translation()[1];
      ob.pose_inLidar[5] = local_t.translation()[2];
      objects.push_back(ob);
    }
    fin_object_file.close();
    frame.objects = objects;
    frames.push_back(frame);

  }
  return frames;
}

