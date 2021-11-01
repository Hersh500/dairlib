//
// Created by hersh on 10/27/21.
//
#include "examples/Cassie/camera_utils.h"

using Eigen::Vector3d;
using Eigen::VectorXd;
using drake::geometry::SceneGraph;
using drake::math::RigidTransform;
using drake::math::RigidTransformd;
using drake::math::RollPitchYaw;
using drake::math::RotationMatrix;

namespace dairlib {
namespace camera {

//  https://github.com/RobotLocomotion/drake/blob/master/examples/manipulation_station/manipulation_station.cc
std::pair <drake::geometry::render::ColorRenderCamera,
drake::geometry::render::DepthRenderCamera>
MakeD415CameraModel(const std::string &renderer_name) {
    // Typical D415 intrinsics for 848 x 480 resolution, note that rgb and
    // depth are slightly different (in both intrinsics and relative to the
    // camera body frame).
    // RGB:
    // - w: 848, h: 480, fx: 616.285, fy: 615.778, ppx: 405.418, ppy: 232.864
    // DEPTH:
    // - w: 848, h: 480, fx: 645.138, fy: 645.138, ppx: 420.789, ppy: 239.13
    const int kHeight = 480;
    const int kWidth = 848;

    // To pose the two sensors relative to the camera body, we'll assume X_BC = I,
    // and select a representative value for X_CD drawn from calibration to define
    // X_BD.
    drake::geometry::render::ColorRenderCamera color_camera{
            {renderer_name,
             {kWidth, kHeight, 616.285, 615.778, 405.418, 232.864} /* intrinsics */,
             {0.01, 3.0} /* clipping_range */,
             {} /* X_BC */},
            false};
    const RigidTransformd X_BD(
            RotationMatrix<double>(RollPitchYaw<double>(
                    -0.19 * M_PI / 180, -0.016 * M_PI / 180, -0.03 * M_PI / 180)),
            Vector3d(0.015, -0.00019, -0.0001));
    drake::geometry::render::DepthRenderCamera depth_camera{
            {renderer_name,
                    {kWidth, kHeight, 645.138, 645.138, 420.789, 239.13} /* intrinsics */,
                    {0.01, 3.0} /* clipping_range */,
                    X_BD},
            {0.1,   2.0} /* depth_range */};
    return {color_camera, depth_camera};
}
};
} // namespace dairlib
