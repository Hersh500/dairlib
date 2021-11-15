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
using drake::geometry::GeometryInstance;
using drake::math::RigidTransform;
using drake::multibody::MultibodyPlant;
using std::make_unique;

namespace dairlib {
    void generateRandomObstacles(MultibodyPlant<double> *plant,
                                 std::pair<double, double> x_lims,
                                 std::pair<double, double> y_lims) {
        if (!plant->geometry_source_is_registered()) {
            return;
        }
        // std::default_random_engine generator;
        std::random_device generator;
        std::uniform_real_distribution<double> x_dist(x_lims.first, x_lims.second);
        std::uniform_real_distribution<double> y_dist(y_lims.first, y_lims.second);
        double x;
        double y;

        for (int i = 0; i < 10; i++) {
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

    // TODO(hersh500): This doesn't work for collisions (results in an error when the collision happens)
    void generateRandomObstacles(SceneGraph<double>& graph,
                                      std::pair<double, double> x_lims,
                                      std::pair<double, double> y_lims) {
        drake::geometry::SourceId terrain_geom_id = graph.RegisterSource("terrain_adder");
        drake::RandomGenerator random_gen = drake::RandomGenerator();
        auto x_dist = drake::schema::Uniform(x_lims.first, x_lims.second);
        auto y_dist = drake::schema::Uniform(y_lims.first, y_lims.second);
        for (int i = 0; i < 10; i++) {
            double x = x_dist.Sample(&random_gen);
            double y = y_dist.Sample(&random_gen);
            // Generate a random pose for this object; maybe based on some stochastic process?
            RigidTransform<double> pose = RigidTransform<double>(drake::math::RollPitchYaw<double>(0, 0, 0),
                                                                 Eigen::Vector3d(x, y, 0));

            drake::geometry::GeometryId box_id = graph.RegisterAnchoredGeometry(terrain_geom_id, make_unique<drake::geometry::GeometryInstance>(
                    pose,make_unique<drake::geometry::Box>(0.5, 0.5, 2), "box_" + std::to_string(i)));
            drake::geometry::PerceptionProperties properties = drake::geometry::PerceptionProperties();
            properties.AddProperty("label", "id", drake::geometry::render::RenderLabel(box_id.get_value()));
            graph.AssignRole(terrain_geom_id, box_id, properties);
            graph.AssignRole(terrain_geom_id, box_id, drake::geometry::IllustrationProperties());
            drake::geometry::ProximityProperties prox_props = drake::geometry::ProximityProperties();
            prox_props.AddProperty(drake::geometry::internal::kMaterialGroup,
                                   drake::geometry::internal::kFriction, drake::multibody::CoulombFriction(.8, .8));
            graph.AssignRole(terrain_geom_id, box_id, prox_props);
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
        double prev_x = 0.0;
        RigidTransform<double> pose = RigidTransform<double>(drake::math::RollPitchYaw<double>(0, 0, 0),
                                                             Eigen::Vector3d(0, 0, -0.05));

        plant->RegisterCollisionGeometry(plant->world_body(), pose, drake::geometry::Box(1, 2, 0.1),
                                         "box_collision_"+ std::to_string(0), drake::multibody::CoulombFriction(0.8, 0.8));

        plant->RegisterVisualGeometry(plant->world_body(), pose, drake::geometry::Box(1, 2, 0.1),
                                      "box_visual_"+std::to_string(0), drake::geometry::IllustrationProperties());

        for (int i = 1; i < 10; i++) {
            double gap = gap_len(generator);
            // Generate a random pose for this object; maybe based on some stochastic process?
            RigidTransform<double> pose = RigidTransform<double>(drake::math::RollPitchYaw<double>(0, 0, 0),
                                                                 Eigen::Vector3d(prev_x + gap + 1, 0, -0.05));

            plant->RegisterCollisionGeometry(plant->world_body(), pose, drake::geometry::Box(1, 2, 0.1),
                                             "box_collision_"+ std::to_string(i), drake::multibody::CoulombFriction(0.8, 0.8));

            // TODO(hersh500): according to drake documentation, this will not work for perception "soon"tm
            plant->RegisterVisualGeometry(plant->world_body(), pose, drake::geometry::Box(1, 2, 0.1),
                                          "box_visual_"+std::to_string(i), drake::geometry::IllustrationProperties());
            prev_x = prev_x + gap + 1;
        }

    }




}

