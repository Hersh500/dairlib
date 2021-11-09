#ifndef DAIRLIB_CAMERA_UTILS_H
#define DAIRLIB_CAMERA_UTILS_H
#include "drake/common/eigen_types.h"
#include "drake/geometry/render/render_engine.h"
#include "drake/math/rigid_transform.h"
#include "drake/multibody/plant/multibody_plant.h"
#include "drake/systems/framework/diagram.h"
using namespace std;

namespace dairlib {
namespace camera {

std::pair<drake::geometry::render::ColorRenderCamera,
        drake::geometry::render::DepthRenderCamera> MakeD415CameraModel(const std::string &renderer_name);

std::pair<drake::geometry::render::ColorRenderCamera,
        drake::geometry::render::DepthRenderCamera> MakeGenericCameraModel(const std::string &renderer_name);
}
}
#endif //DAIRLIB_CAMERA_UTILS_H
