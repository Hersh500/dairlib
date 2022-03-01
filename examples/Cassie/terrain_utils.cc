//
// Created by hersh on 11/8/21.
//
#include <random>
#include <drake/geometry/shape_specification.h>
#include "examples/Cassie/terrain_utils.h"
#include "drake/geometry/scene_graph.h"
#include <drake/common/schema/stochastic.h>
#include <drake/multibody/plant/multibody_plant.h>


using drake::geometry::SceneGraph;
using drake::geometry::SourceId;
using drake::geometry::HalfSpace;
using drake::geometry::GeometryInstance;
using drake::math::RigidTransform;
using drake::multibody::MultibodyPlant;
using std::make_unique;

namespace dairlib {
    void generateRandomObstacles(MultibodyPlant<double> *plant,
                                 std::pair<double, double> x_lims,
                                 std::pair<double, double> y_lims,
                                 unsigned int num_obs) {
        if (!plant->geometry_source_is_registered()) {
            return;
        }
        // std::default_random_engine generator;
        std::random_device generator;
        std::uniform_real_distribution<double> x_dist(x_lims.first, x_lims.second);
        std::uniform_real_distribution<double> y_dist(y_lims.first, y_lims.second);
        double x;
        double y;
        for (int i = 0; i < num_obs; i++) {
            x = x_dist(generator);
            y = y_dist(generator);
            // Generate a random pose for this object; maybe based on some stochastic process?
            RigidTransform<double> pose = RigidTransform<double>(drake::math::RollPitchYaw<double>(0, 0, 0),
                    Eigen::Vector3d(x, y, 0));

            plant->RegisterCollisionGeometry(plant->world_body(), pose, drake::geometry::Box(0.5, 0.5, 2),
                                            "box_collision_"+ std::to_string(i), drake::multibody::CoulombFriction(0.8, 0.8));

            // TODO(hersh500): according to drake documentation, this will not work for perception "soon"tm
            plant->RegisterVisualGeometry(plant->world_body(), pose, drake::geometry::Box(0.5, 0.5, 2),
                                         "box_visual_"+std::to_string(i), drake::geometry::IllustrationProperties());
        }
    }

    // For now, assume that the gaps go down to infinity. Eventually, maybe want to structure this like a pallet test.
    void generateRandomGaps(MultibodyPlant<double> *plant,
                            std::pair<double, double> gap_lims) {
        if (!plant->geometry_source_is_registered()) {
            return;
        }
        std::random_device generator;
        std::uniform_real_distribution<double> gap_len(gap_lims.first, gap_lims.second);
        std::uniform_real_distribution<double> color(0, 1);
        double prev_x = 0.0;
        RigidTransform<double> pose = RigidTransform<double>(drake::math::RollPitchYaw<double>(0, 0, 0),
                                                             Eigen::Vector3d(0, 0, -0.05));

        plant->RegisterCollisionGeometry(plant->world_body(), pose, drake::geometry::Box(1, 2, 0.1),
                                         "box_collision_"+ std::to_string(0), drake::multibody::CoulombFriction(0.8, 0.8));
//        drake::geometry::IllustationProperties properties = drake::geometry::IllustrationProperties();
//        properties.("phong", "diffuse", )
        drake::geometry::IllustrationProperties p = drake::geometry::MakePhongIllustrationProperties(Eigen::Vector4d(color(generator),
                                                                                                                     color(generator),
                                                                                                                     color(generator),
                                                                                                                     1));
        plant->RegisterVisualGeometry(plant->world_body(), pose, drake::geometry::Box(1, 2, 0.1),
                                      "box_visual_"+std::to_string(0), p);

        for (int i = 1; i < 10; i++) {
            double gap = gap_len(generator);
            // Generate a random pose for this object; maybe based on some stochastic process?
            RigidTransform<double> pose = RigidTransform<double>(drake::math::RollPitchYaw<double>(0, 0, 0),
                                                                 Eigen::Vector3d(prev_x + gap + 1, 0, -0.05));

            plant->RegisterCollisionGeometry(plant->world_body(), pose, drake::geometry::Box(1, 2, 0.1),
                                             "box_collision_"+ std::to_string(i), drake::multibody::CoulombFriction(0.8, 0.8));

            // TODO(hersh500): according to drake documentation, this will not work for perception "soon"tm
            drake::geometry::IllustrationProperties p = drake::geometry::MakePhongIllustrationProperties(Eigen::Vector4d(color(generator),
                                                                                                                         color(generator),
                                                                                                                         color(generator),
                                                                                                                         1));

            plant->RegisterVisualGeometry(plant->world_body(), pose, drake::geometry::Box(1, 2, 0.1),
                                          "box_visual_"+std::to_string(i), p);
            prev_x = prev_x + gap + 1;
        }

    }

    void generateFixedSteps(MultibodyPlant<double> *plant) {
      if (!plant->geometry_source_is_registered()) {
        return;
      }
      std::random_device generator;
      // std::uniform_real_distribution<double> step_height(step_lims.first, step_lims.second);
      double prev_z = -0.05;
      double prev_x = 1.0;
      RigidTransform<double> pose = RigidTransform<double>(drake::math::RollPitchYaw<double>(0, 0, 0),
                                                           Eigen::Vector3d(0, 0, prev_z));

      plant->RegisterCollisionGeometry(plant->world_body(), pose, drake::geometry::Box(1, 2, 0.1),
                                       "box_collision_"+ std::to_string(0), drake::multibody::CoulombFriction(0.8, 0.8));
      plant->RegisterVisualGeometry(plant->world_body(), pose, drake::geometry::Box(1, 2, 0.1),
                                    "box_visual_"+std::to_string(0), drake::geometry::IllustrationProperties());

      for (int i = 1; i < 10; i++) {
        RigidTransform<double> pose = RigidTransform<double>(drake::math::RollPitchYaw<double>(0, 0, 0),
                                                             Eigen::Vector3d(prev_x + 1, 0, prev_z + 0.1));

        plant->RegisterCollisionGeometry(plant->world_body(), pose, drake::geometry::Box(1, 2, 0.1),
                                         "box_collision_"+ std::to_string(i), drake::multibody::CoulombFriction(0.8, 0.8));

        plant->RegisterVisualGeometry(plant->world_body(), pose, drake::geometry::Box(1, 2, 0.1),
                                      "box_visual_"+std::to_string(i), drake::geometry::IllustrationProperties());
        prev_x = prev_x + 1;
        prev_z = prev_z + 0.1;
      }

    }

    void generateReferenceTerrain(MultibodyPlant<double> *plant) {
      if (!plant->geometry_source_is_registered()) {
        return;
      }
      RigidTransform<double> pose = RigidTransform<double>(drake::math::RollPitchYaw<double>(0, 0, 0),
                                                           Eigen::Vector3d(0.1, 0, 0));

//      plant->RegisterCollisionGeometry(plant->world_body(), pose, drake::geometry::Box(0.05, 0.05, 0.05),
//                                       "box_collision_" + std::to_string(0),
//                                       drake::multibody::CoulombFriction(0.8, 0.8));
      plant->RegisterVisualGeometry(plant->world_body(), pose, drake::geometry::Box(0.05, 0.05, 0.1),
                                    "box_visual_" + std::to_string(0), drake::geometry::IllustrationProperties());

      pose = RigidTransform<double>(drake::math::RollPitchYaw<double>(0.05, 0.05, 0.05),
                                                           Eigen::Vector3d(0.2, 0, 0));

//      plant->RegisterCollisionGeometry(plant->world_body(), pose, drake::geometry::Box(0.05, 0.05, 0.05),
//                                       "box_collision_" + std::to_string(1),
//                                       drake::multibody::CoulombFriction(0.8, 0.8));
      plant->RegisterVisualGeometry(plant->world_body(), pose, drake::geometry::Box(0.05, 0.05, 0.2),
                                    "box_visual_" + std::to_string(1), drake::geometry::IllustrationProperties());

      pose = RigidTransform<double>(drake::math::RollPitchYaw<double>(0, 0, 0),
                                                           Eigen::Vector3d(0.3, 0, 0));

//      plant->RegisterCollisionGeometry(plant->world_body(), pose, drake::geometry::Box(0.05, 0.05, 0.05),
//                                       "box_collision_" + std::to_string(2),
//                                       drake::multibody::CoulombFriction(0.8, 0.8));
      plant->RegisterVisualGeometry(plant->world_body(), pose, drake::geometry::Box(0.05, 0.05, 0.3),
                                    "box_visual_" + std::to_string(2), drake::geometry::IllustrationProperties());

      pose = RigidTransform<double>(drake::math::RollPitchYaw<double>(0, 0, 0),
                                    Eigen::Vector3d(0.4, 0.0, 0));

      plant->RegisterVisualGeometry(plant->world_body(), pose, drake::geometry::Box(0.05, 0.05, 0.4),
                                    "box_visual_" + std::to_string(3), drake::geometry::IllustrationProperties());

      pose = RigidTransform<double>(drake::math::RollPitchYaw<double>(0, 0, 0),
                                    Eigen::Vector3d(0.5, 0.0, 0));

      plant->RegisterVisualGeometry(plant->world_body(), pose, drake::geometry::Box(0.05, 0.05, 0.5),
                                    "box_visual_" + std::to_string(4), drake::geometry::IllustrationProperties());
    }

    void generateFixedObstacleCourse(MultibodyPlant<double> *plant) {
      if (!plant->geometry_source_is_registered()) {
        return;
      }

      Eigen::MatrixXd locs(8,2);
      locs << 1.5, 1.5,
          1.5, -1.5,
          -1.5, 1.5,
          -1.5, -1.5,
          2.5, 0,
          0, 2.5,
          -2.5, 0,
          0, -2.5;

      double width = 0.5;
      double height = 1;
      for (int i = 0; i < locs.rows(); i++) {
        double x = locs(i, 0);
        double y = locs(i, 1);

        // Generate a random pose for this object; maybe based on some stochastic process?
        RigidTransform<double> pose = RigidTransform<double>(drake::math::RollPitchYaw<double>(0, 0, 0),
                                                             Eigen::Vector3d(x, y, height/2));

        plant->RegisterCollisionGeometry(plant->world_body(), pose, drake::geometry::Box(width, width, height),
                                         "box_collision_"+ std::to_string(i), drake::multibody::CoulombFriction(0.8, 0.8));

        plant->RegisterVisualGeometry(plant->world_body(), pose, drake::geometry::Box(width, width, height),
                                      "box_visual_"+std::to_string(i), drake::geometry::IllustrationProperties());
      }
    }
}
