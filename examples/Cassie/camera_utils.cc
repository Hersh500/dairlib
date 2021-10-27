//
// Created by hersh on 10/27/21.
//

#include "drake/common/find_resource.h"
#include "drake/geometry/render/render_engine_vtk_factory.h"
#include "drake/manipulation/schunk_wsg/schunk_wsg_constants.h"
#include "drake/manipulation/schunk_wsg/schunk_wsg_position_controller.h"
#include "drake/math/rigid_transform.h"
#include "drake/math/rotation_matrix.h"
#include "drake/multibody/parsing/parser.h"
#include "drake/multibody/tree/prismatic_joint.h"
#include "drake/multibody/tree/revolute_joint.h"
#include "drake/perception/depth_image_to_point_cloud.h"
#include "drake/systems/controllers/inverse_dynamics_controller.h"
#include "drake/systems/framework/diagram_builder.h"
#include "drake/systems/primitives/adder.h"
#include "drake/systems/primitives/constant_value_source.h"
#include "drake/systems/primitives/constant_vector_source.h"
#include "drake/systems/primitives/demultiplexer.h"
#include "drake/systems/primitives/discrete_derivative.h"
#include "drake/systems/primitives/linear_system.h"
#include "drake/systems/primitives/matrix_gain.h"
#include "drake/systems/primitives/pass_through.h"
#include "drake/systems/sensors/rgbd_sensor.h"

using Eigen::Vector3d;
using Eigen::VectorXd;
using drake::geometry::SceneGraph;
using drake::geometry::render::MakeRenderEngineVtk;
using drake::geometry::render::RenderEngineVtkParams;
using drake::math::RigidTransform;
using drake::math::RigidTransformd;
using drake::math::RollPitchYaw;
using drake::math::RotationMatrix;

namespace dairlib {
namespace camera {

//  https://github.com/RobotLocomotion/drake/blob/master/examples/manipulation_station/manipulation_station.cc
std::pair <geometry::render::ColorRenderCamera,
geometry::render::DepthRenderCamera>
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
    geometry::render::ColorRenderCamera color_camera{
            {renderer_name,
             {kWidth, kHeight, 616.285, 615.778, 405.418, 232.864} /* intrinsics */,
             {0.01, 3.0} /* clipping_range */,
             {} /* X_BC */},
            false};
    const RigidTransformd X_BD(
            RotationMatrix<double>(RollPitchYaw<double>(
                    -0.19 * M_PI / 180, -0.016 * M_PI / 180, -0.03 * M_PI / 180)),
            Vector3d(0.015, -0.00019, -0.0001));
    geometry::render::DepthRenderCamera depth_camera{
            {renderer_name,
                    {kWidth, kHeight, 645.138, 645.138, 420.789, 239.13} /* intrinsics */,
                    {0.01, 3.0} /* clipping_range */,
                    X_BD},
            {0.1,   2.0} /* depth_range */};
    return {color_camera, depth_camera};
}

// Need to add the camera and weld it to the appropriate frame.

void RegisterRgbdSensor(
        const std::string& name, const multibody::Frame<T>& parent_frame,
        const RigidTransform<double>& X_PC,
        const geometry::render::ColorRenderCamera& color_camera,
        const geometry::render::DepthRenderCamera& depth_camera) {
    CameraInformation info;
    info.parent_frame = &parent_frame;
    info.X_PC = X_PC;
    info.depth_camera = depth_camera;
    info.color_camera = color_camera;

    camera_information_[name] = info;
}

/*
    // This blurb should probably be called in the main file
    auto camera = builder.template AddSystem<systems::sensors::RgbdSensor>(
            parent_body_id.value(), X_PC, info.color_camera, info.depth_camera);
    builder.Connect(scene_graph_->get_query_output_port(),
                    camera->query_object_input_port());
*/

};
}
} // namespace dairlib
