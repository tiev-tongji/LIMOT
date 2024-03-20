#pragma once
#include <iostream>
#include <string>
#include <gtsam/geometry/Pose2.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/concepts.h>
#include <gtsam/base/concepts.h>
#include <gtsam/inference/Key.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/GaussNewtonOptimizer.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/nonlinear/Values.h>

#include <gtsam/base/Matrix.h>
#include <gtsam/base/Vector.h>
#include <gtsam/base/Testable.h>
#include <gtsam/base/Lie.h>
#include <gtsam/nonlinear/NonlinearFactor.h>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/Dense>
using namespace gtsam;
using namespace std;
namespace factorgraph
{
  Matrix6 Adj(gtsam::Pose3 T)
  {
    Matrix3 R = T.rotation().matrix();
    Matrix3 A = skewSymmetric(T.x(), T.y(), T.z()) * R;
    Matrix6 adj;
    adj.block(0, 0, 3, 3) = R;
    adj.block(3, 3, 3, 3) = R;
    adj.block(0, 3, 3, 3) = A;
    adj.block(3, 0, 3, 3) = Matrix3::Zero(3, 3);
    return adj;
  }

  // Define the ternary factor
  class BTW3Factor : public gtsam::NoiseModelFactor3<gtsam::Pose3, gtsam::Pose3, gtsam::Pose3>
  {
  private:
    Pose3 T_;

  public:
    BTW3Factor(gtsam::Key poseKey1, gtsam::Key poseKey2, gtsam::Key poseKey3, const gtsam::Pose3 T, gtsam::SharedNoiseModel model) : gtsam::NoiseModelFactor3<gtsam::Pose3, gtsam::Pose3, gtsam::Pose3>(model, poseKey1, poseKey2, poseKey3), T_(T) {}
    // error function
    gtsam::Vector evaluateError(const gtsam::Pose3 &p1, const gtsam::Pose3 &p2, const gtsam::Pose3 &p3, boost::optional<gtsam::Matrix &> H1 = boost::none, boost::optional<gtsam::Matrix &> H2 = boost::none, boost::optional<gtsam::Matrix &> H3 = boost::none) const
    {
      gtsam::Pose3 error_ = p1.inverse() * p2 * p3.inverse();
      Matrix6 J;
      Vector6 log = Pose3::Logmap(error_);
      Vector3 w{0, 0, 0};
      w << log(0), log(1), log(2);
      Vector3 t{0, 0, 0};
      t << log(3), log(4), log(5);
      J.block(0, 0, 3, 3) = skewSymmetric(log(0), log(1), log(2));
      J.block(0, 3, 3, 3) = skewSymmetric(log(3), log(4), log(5));
      J.block(3, 0, 3, 3) = Matrix3::Zero(3, 3);
      J.block(3, 3, 3, 3) = skewSymmetric(log(0), log(1), log(2));
      J = J * 0.5 + Matrix6::Identity();

      if (H1)
        *H1 = ((-1) * J * Adj((p2 * p3.inverse()).inverse()));
      if (H2)
        *H2 = (J * Adj((p2 * p3.inverse()).inverse()));
      if (H3)
        *H3 = ((-1) * J);
      // return error vector
      return log;
    }
  };

  class FactorGraph
  {
  public:
    NonlinearFactorGraph gtSAMgraph2;
    Values initialEstimate2;
    Values result;
    gtsam::noiseModel::Gaussian::shared_ptr updatenoise;
    int f_id = 0; // factor id (edge id)
    int key_id = 0; // state id (node id)
    void AddBTW2factor(int id1, int id2, Pose3 p, double noiseScore)
    {

      gtsam::Vector Vector6(6);
      Vector6 << noiseScore, noiseScore, noiseScore, noiseScore, noiseScore, noiseScore;
      noiseModel::Diagonal::shared_ptr Noise = noiseModel::Diagonal::Variances(Vector6);
      gtSAMgraph2.add(BetweenFactor<Pose3>(id1, id2, p, Noise));
      f_id++;
    }
    void AddRobustBTW2factor(int id1, int id2, Pose3 p, double robustNoiseScore, double rubostNum)
    {
      gtsam::Vector robustNoiseVector6(6);
      robustNoiseVector6 << robustNoiseScore, robustNoiseScore, robustNoiseScore, robustNoiseScore, robustNoiseScore, robustNoiseScore;
      noiseModel::Base::shared_ptr robustConstraintNoise;
      robustConstraintNoise = gtsam::noiseModel::Robust::Create(gtsam::noiseModel::mEstimator::Cauchy::Create(rubostNum), // Cauchy is empirically enough.
                                                                gtsam::noiseModel::Diagonal::Variances(robustNoiseVector6));
      gtSAMgraph2.add(BetweenFactor<Pose3>(id1, id2, p, robustConstraintNoise));
      f_id++;
    }
    void AddBTW3factor(int id1, int id2, int id3, double noiseScore)
    {
      gtsam::Pose3 p(gtsam::Rot3::RzRyRx(0.0, 0.0, 0.0), gtsam::Point3(0.0, 0.0, 0.0));
      gtsam::Vector Vector6(6);
      Vector6 << noiseScore, noiseScore, noiseScore, noiseScore, noiseScore, noiseScore;
      noiseModel::Diagonal::shared_ptr Noise = noiseModel::Diagonal::Variances(Vector6);
      gtSAMgraph2.add(BTW3Factor(id1, id2, id3, p, Noise));
      f_id++;
    }
    void AddRobustBTW3factor(int id1, int id2, int id3, double robustNoiseScore, double rubostNum)
    {
      gtsam::Pose3 p(gtsam::Rot3::RzRyRx(0.0, 0.0, 0.0), gtsam::Point3(0.0, 0.0, 0.0));
      gtsam::Vector robustNoiseVector6(6);
      robustNoiseVector6 << robustNoiseScore, robustNoiseScore, robustNoiseScore, robustNoiseScore, robustNoiseScore, robustNoiseScore;
      noiseModel::Base::shared_ptr robustConstraintNoise;
      robustConstraintNoise = gtsam::noiseModel::Robust::Create(gtsam::noiseModel::mEstimator::Cauchy::Create(rubostNum),
                                                                gtsam::noiseModel::Diagonal::Variances(robustNoiseVector6));
      gtSAMgraph2.add(BTW3Factor(id1, id2, id3, p, robustConstraintNoise));
      f_id++;
    }

    void AddPriorFactor(int id, Pose3 p, double noiseScore)
    {

      gtsam::Vector Vector6(6);
      Vector6 << noiseScore, noiseScore, noiseScore, noiseScore, noiseScore, noiseScore;
      noiseModel::Diagonal::shared_ptr Noise = noiseModel::Diagonal::Variances(Vector6);
      gtSAMgraph2.add(PriorFactor<Pose3>(id, p, Noise));
      f_id++;
    }

    void AddRobustPriorFactor(int id, Pose3 p, double robustNoiseScore, double rubostNum)
    {

      gtsam::Vector robustNoiseVector6(6);
      robustNoiseVector6 << robustNoiseScore, robustNoiseScore, robustNoiseScore, robustNoiseScore, robustNoiseScore, robustNoiseScore;
      noiseModel::Base::shared_ptr robustConstraintNoise;
      robustConstraintNoise = gtsam::noiseModel::Robust::Create(gtsam::noiseModel::mEstimator::Cauchy::Create(rubostNum), 
                                                                gtsam::noiseModel::Diagonal::Variances(robustNoiseVector6));
      gtSAMgraph2.add(PriorFactor<Pose3>(id, p, robustConstraintNoise));
      f_id++;
    }

    void AddMarginalPrior(int id, Pose3 p, gtsam::Marginals marginals)
    {
      updatenoise = gtsam::noiseModel::Gaussian::Covariance(marginals.marginalCovariance(id));
      gtSAMgraph2.add(PriorFactor<Pose3>(id, p, updatenoise));
      f_id++;
    }
    void AddRobustMarginalPrior(int id, Pose3 p, gtsam::Marginals marginals, double rubostNum)
    {
      noiseModel::Base::shared_ptr robustupdateNoise;
      robustupdateNoise = gtsam::noiseModel::Robust::Create(gtsam::noiseModel::mEstimator::Cauchy::Create(rubostNum), 
                                                            gtsam::noiseModel::Gaussian::Covariance(marginals.marginalCovariance(id)));
      gtSAMgraph2.add(PriorFactor<Pose3>(id, p, robustupdateNoise));
      f_id++;
    }
    void setinitialEstimate(int id, Pose3 p)
    {
      initialEstimate2.insert(id, p);
    }
    void Removefactor(int id)
    {
      gtSAMgraph2.remove(id);
    }
    void Removekey(int id)
    {
      initialEstimate2.erase(id);
    }

    void StartOptimiz(int factor_num)
    {
      int iterations;
      if (factor_num >= 100)
        iterations = 70;
      else
        iterations = 30;
      LevenbergMarquardtParams parameters;
      parameters.maxIterations = iterations;
      LevenbergMarquardtOptimizer optimizer(gtSAMgraph2, initialEstimate2, parameters);
      result = optimizer.optimize();
    }
  };
}
