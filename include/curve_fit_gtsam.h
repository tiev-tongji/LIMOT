#pragma once
#include <gtsam/base/Vector.h>
#include <gtsam/slam/dataset.h>

#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>

#include <gtsam/nonlinear/DoglegOptimizer.h>
// #include <gtsam/nonlinear/Marginals.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/ISAM2.h>

class curvfitFactor : public gtsam::NoiseModelFactor1<gtsam::Vector4>
{
    double mt, mxy;

public:
    curvfitFactor(gtsam::Key key, double t, double xy, gtsam::SharedNoiseModel model)
        : gtsam::NoiseModelFactor1<gtsam::Vector4>(model, key), mt(t), mxy(xy) {}

    virtual ~curvfitFactor()
    {
    }
    gtsam::Vector evaluateError(const gtsam::Vector4 &abcd, boost::optional<gtsam::Matrix &> H = boost::none) const
    {
        auto val = abcd[0] * mt * mt * mt + abcd[1] * mt * mt + abcd[2] * mt + abcd[3];
        if (H)
        {
            gtsam::Matrix Jac = gtsam::Matrix::Zero(1, 4);
            Jac << mt * mt * mt, mt * mt, mt, 1;
            (*H) = Jac;
        }
        return gtsam::Vector1(val - mxy);
    }
};

// Computational model of the cost function

class CurveFit
{
private:
    double abcd[4] = {0, 0, 0, 0}; // Parameters to be estimated
    NonlinearFactorGraph graph;
    Values initial;

public:
    double df = 0;
    void Fitting(std::vector<double> t, std::vector<double> xy)
    {
        double noiseScore = 1e-4;
        gtsam::Vector Vector1(1);
        Vector1 << noiseScore;
        noiseModel::Diagonal::shared_ptr Noise = noiseModel::Diagonal::Variances(Vector1);

        if (t.size() == xy.size())
        {
            for (int i = 0; i < xy.size(); i++)
            {
                graph.emplace_shared<curvfitFactor>(0, t[i] - t[0], xy[i], Noise);
            }
            df = t[0];
            initial.insert(0, gtsam::Vector4(0.0, 0.0, 0.0, 0.0));
            // graph.print();
        }
        else
        {
            std::cout << "data error!" << std::endl;
        }
        gtsam::LevenbergMarquardtParams parameters;
        parameters.maxIterations = 30;
        gtsam::LevenbergMarquardtOptimizer opt(graph, initial, parameters);
        Values results;
        results = opt.optimize();
        gtsam::Vector4 v = results.at<gtsam::Vector4>(0);
        for (int i = 0; i < 4; i++) 
        {
            abcd[i] = v[i];
        }
        graph.resize(0);
        initial.clear();
    }
    double Predict(double t)
    {
        t = t - df;
        return abcd[0] * t * t * t + abcd[1] * t * t + abcd[2] * t + abcd[3];
    }

    std::vector<double> GetParams()
    {
        return std::vector<double>(abcd, abcd + 4);
    }
};
