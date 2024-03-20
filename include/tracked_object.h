#pragma once
#include <structs.h>
#include <curve_fit_gtsam.h>
#include <successive_shortest_path.h>
#include <utility>
void DataAssociationBySSP(std::vector<std::vector<double>> _matrix, std::vector<int>& _order, std::vector<int>& _reverse_order) {
  std::unordered_map<int, int> direct_assignment;
  std::unordered_map<int, int> reverse_assignment;
  assignment_problem::MaximizeLinearAssignment(_matrix, &direct_assignment, &reverse_assignment);
  for (int i = 0; i < _order.size(); i++) {
    if (direct_assignment.find(i) != direct_assignment.end())
      _order[i] = direct_assignment[i];
  }
  for (int i = 0; i < _reverse_order.size(); i++) {
    if (reverse_assignment.find(i) != reverse_assignment.end())
      _reverse_order[i] = reverse_assignment[i];
  }
}

class TrackedObstacle {
private:
  std::vector<double> track_time;

  CurveFit x_t_fit;
  CurveFit y_t_fit;
  int kk;

public:
  std::vector<double> track_utm_x;
  std::vector<double> track_utm_y;
  int missed;
  int num_observations;
  std::pair<int, int> last_obs;
  int obj_id = -2;

  TrackedObstacle() {
    num_observations = 0;
    missed = 0;
  }

  bool Predict(double time, std::vector<double>& predict_pos) {
    if (num_observations >= 5) {
      predict_pos.push_back(x_t_fit.Predict(time));
      predict_pos.push_back(y_t_fit.Predict(time));
      return true;
    }
    else {
      predict_pos.push_back(track_utm_x[num_observations - 1]);
      predict_pos.push_back(track_utm_y[num_observations - 1]);
      return false;
    }
  }

  void Add(double frame_num, LidarSLAMObject object) {

    track_time.push_back(frame_num);
    track_utm_x.push_back(object.optimize_t[3]);
    track_utm_y.push_back(object.optimize_t[4]);
    if (obj_id == -2)
      obj_id = object.object_id;
    else if (obj_id != object.object_id) {
      obj_id = object.object_id;
    }

    if (num_observations >= 4) {
      int num = num_observations > 10 ? 10 : num_observations;
      x_t_fit.Fitting(
        std::vector<double>(track_time.end() - num, track_time.end()),
        std::vector<double>(track_utm_x.end() - num, track_utm_x.end()));
      y_t_fit.Fitting(
        std::vector<double>(track_time.end() - num, track_time.end()),
        std::vector<double>(track_utm_y.end() - num, track_utm_y.end()));
    }
  };

  bool IsDynamic(float vel_threshold, std::queue<std::vector<double>>& dynamBoxBuf, int flow, LidarSLAMObject object, Eigen::Affine3f ego_T) {
    int num = num_observations > 5 ? 5 : num_observations;
    double x_del = track_utm_x[num_observations - 1] - track_utm_x[num_observations - num];
    double y_del = track_utm_y[num_observations - 1] - track_utm_y[num_observations - num];
    double d_del = sqrt(x_del * x_del + y_del * y_del);
    double vel = d_del / (num - 1) * 10;
    if (vel > vel_threshold && num >= 2) ////过滤动态点的阈值其实可以再调大一点。
    {
      double pre_x;
      double pre_y;
      if (num >= 4) {
        pre_x = x_t_fit.Predict(flow + 1);
        pre_y = y_t_fit.Predict(flow + 1);
      }
      else {
        pre_x = object.optimize_t[3];
        pre_y = object.optimize_t[4];
      }

      std::vector<double> v(9, 0); // x,y,z,yaw_local,l,w,h,global_yaw，score
      v[0] = pre_x;
      v[1] = pre_y;
      v[2] = object.optimize_t[5];
      v[7] = object.optimize_t[2];
      v[8] = object.score;
      v[3] = object.rotation_zyx[0];
      for (int i = 0; i < 3; i++) {
        v[i + 4] = object.measure_lwh[i];
      }
      dynamBoxBuf.push(v);
    }
    if (vel > vel_threshold)
      return true;
    else
      return false;
  }
};

class Tracker {
private:
  std::vector<TrackedObstacle> tracked_object;
  int id;

public:
  Tracker() { id = 0; }

  void Update_TrackUtmXY(int id, float In_x, float In_y) {
    for (auto obj : tracked_object) {
      if (obj.obj_id == id) {
        obj.track_utm_x[obj.num_observations - 1] = In_x;
        obj.track_utm_y[obj.num_observations - 1] = In_y;
      }
    }
  }

  void AssociateObjects(std::vector<LidarSLAMFrame, Eigen::aligned_allocator<LidarSLAMFrame>>& frames, int flow, float vel_threshold, std::queue<std::vector<double>>& dynamBoxBuf) {
    if (dynamBoxBuf.size() != 0) {
      std::cout << "wrong! dynamBoxBuf != 0 , size is : " << dynamBoxBuf.size() << std::endl;
    }
    int tracked_size = tracked_object.size();
    int detected_size = frames[flow].objects.size();
    std::vector<int> reverse_order = std::vector<int>(detected_size, -1);
    std::vector<int> order = std::vector<int>(tracked_size, -1);
    std::cout << "tracked size: " << tracked_size << "      detected num:" << detected_size << std::endl;
    if (tracked_size != 0 && detected_size != 0) {
      std::vector<std::vector<double>> matrix(tracked_object.size(), std::vector<double>(frames[flow].objects.size(), 0));
      double predict_time = frames[flow].frame_id;
      for (int n = 0; n < tracked_size; n++) {
        std::vector<double> predict_pos;
        bool status = tracked_object[n].Predict(predict_time, predict_pos);
        Eigen::Vector3f local_xyz = frames[flow].optimize_pose.inverse() * Eigen::Vector3f(predict_pos[0], predict_pos[1], 0);

        for (int m = 0; m < detected_size; m++) {
          double distance =
            sqrt(pow(predict_pos[0] - frames[flow].objects[m].optimize_t[3], 2) +
              pow(predict_pos[1] - frames[flow].objects[m].optimize_t[4], 2));
          if ((status && distance < 2) || (!status && distance < 3.5))
            matrix[n][m] = (100.0 - distance) / 100.0; // alpha is 100
        }
      }
      DataAssociationBySSP(matrix, order, reverse_order);
    }
    for (int n = 0; n < order.size(); n++) {
      if (order[n] != -1) {
        int de_index = order[n];
        int ob_frame = tracked_object[n].last_obs.first;
        int ob_index = tracked_object[n].last_obs.second;
        frames[flow].objects[de_index].object_id = frames[ob_frame].objects[ob_index].object_id;
        frames[flow].objects[de_index].associated = true;
        frames[flow].objects[de_index].associated_object_last = tracked_object[n].last_obs;
        if (frames[ob_frame].objects[ob_index].associated)
          frames[flow].objects[de_index].continuous_associated = true;
        else
          frames[flow].objects[de_index].continuous_associated = false;
        frames[ob_frame].objects[ob_index].associated_object_next = std::make_pair(flow, de_index);
        tracked_object[n].num_observations++;

        tracked_object[n].missed = 0;
        if (tracked_object[n].num_observations >= 5) {
          frames[flow].objects[de_index].initialized = true;
        }
        else
          frames[flow].objects[de_index].initialized = false;
        tracked_object[n].Add(frames[flow].frame_id, frames[flow].objects[de_index]);
        tracked_object[n].last_obs = std::make_pair(flow, de_index);
        Eigen::Affine3f ego_T = frames[flow].optimize_pose;
        frames[flow].objects[de_index].dynamic = tracked_object[n].IsDynamic(vel_threshold, dynamBoxBuf, frames[flow].frame_id, frames[flow].objects[de_index], ego_T);
      }
      else {
        tracked_object[n].missed++;
      }
    }

    for (int n = 0; n < reverse_order.size(); n++) {
      if (reverse_order[n] == -1) {
        std::pair<int, int> cur_obj(flow, n);
        frames[flow].objects[n].object_id = id++;
        frames[flow].objects[n].associated = false;
        frames[flow].objects[n].continuous_associated = false;
        frames[flow].objects[n].initialized = false;
        TrackedObstacle ob;
        ob.num_observations++;
        std::vector<double> v(9, 0); // x,y,z,yaw_local,l,w,h,global_yaw，score
        v[0] = frames[flow].objects[n].optimize_t[3];
        v[1] = frames[flow].objects[n].optimize_t[4];
        v[2] = frames[flow].objects[n].optimize_t[5];
        v[7] = frames[flow].objects[n].optimize_t[2];
        v[8] = frames[flow].objects[n].score;
        v[3] = frames[flow].objects[n].rotation_zyx[0];
        for (int i = 0; i < 3; i++) {
          v[i + 4] = frames[flow].objects[n].measure_lwh[i];
        }
        dynamBoxBuf.push(v);
        ob.missed = 0;
        ob.Add(frames[flow].frame_id, frames[flow].objects[n]);
        ob.last_obs = cur_obj;
        tracked_object.push_back(ob);
        frames[flow].objects[n].dynamic = false;
      }
    }
    for (int k = tracked_object.size() - 1; k >= 0; k--) {
      int ob_frame = tracked_object[k].last_obs.first;
      int ob_index = tracked_object[k].last_obs.second;
      if ((frames[ob_frame].objects[ob_index].initialized && tracked_object[k].missed > 1) ||
        (!frames[ob_frame].objects[ob_index].initialized && tracked_object[k].missed > 0)) {
        tracked_object.erase(tracked_object.begin() + k);
      }
    }

    for (int k = 0; k < tracked_object.size(); k++) { // missing detection, so create temporary objects
      if (tracked_object[k].missed != 0) {
        int ob_frame = tracked_object[k].last_obs.first;
        int ob_index = tracked_object[k].last_obs.second;
        LidarSLAMObject temp_object;
        LidarSLAMObject latest_object = frames[ob_frame].objects[ob_index];
        temp_object.frame_id = frames[flow].frame_id;
        temp_object.obj_status = 2;
        temp_object.object_id = latest_object.object_id;
        temp_object.type = latest_object.type;
        temp_object.measure_lwh = latest_object.measure_lwh;
        temp_object.rotation_zyx = latest_object.rotation_zyx;
        temp_object.pose_inLidar[2] = latest_object.pose_inLidar[2];
        std::vector<double> predict_pos;
        bool status = tracked_object[k].Predict(frames[flow].frame_id, predict_pos);
        Eigen::Vector3f local_xyz = Eigen::Vector3f::Zero();
        local_xyz = frames[flow].optimize_pose.inverse() * Eigen::Vector3f(predict_pos[0], predict_pos[1], 0);
        for (int i = 0; i < 3; i++) {
          temp_object.local_xyz[i] = local_xyz(i);
          temp_object.pose_inLidar[3 + i] = local_xyz(i);
        }

        temp_object.local_xyz[2] = latest_object.local_xyz[2];
        local_xyz(2) = latest_object.local_xyz[2];
        temp_object.pose_inLidar[5] = latest_object.local_xyz[2];
        Eigen::Vector3f v = frames[flow].optimize_pose * local_xyz;
        for (int i = 0; i < 3; i++)
          temp_object.optimize_t[i + 3] = v(i);
        frames[flow].objects.push_back(temp_object);
        frames[flow].objects[frames[flow].objects.size() - 1].associated = true;
        frames[flow].objects[frames[flow].objects.size() - 1].associated_object_last = std::make_pair(ob_frame, ob_index);
        if (frames[ob_frame].objects[ob_index].associated) {
          frames[flow].objects[frames[flow].objects.size() - 1].continuous_associated = true;
        }
        else {
          frames[flow].objects[frames[flow].objects.size() - 1].continuous_associated = false;
        }
        frames[ob_frame].objects[ob_index].associated_object_next = std::make_pair(flow, frames[flow].objects.size() - 1);
        tracked_object[k].num_observations++;
        if (tracked_object[k].num_observations >= 5)
          frames[flow].objects[frames[flow].objects.size() - 1].initialized = true;
        else
          frames[flow].objects[frames[flow].objects.size() - 1].initialized = false;
        tracked_object[k].Add(frames[flow].frame_id, frames[flow].objects[frames[flow].objects.size() - 1]);
        tracked_object[k].last_obs = std::make_pair(flow, frames[flow].objects.size() - 1);
        Eigen::Affine3f ego_T = frames[flow].optimize_pose;
        frames[flow].objects[frames[flow].objects.size() - 1].dynamic = tracked_object[k].IsDynamic(vel_threshold, dynamBoxBuf, frames[flow].frame_id, frames[flow].objects[frames[flow].objects.size() - 1], ego_T);
      }
    }
  }
};
