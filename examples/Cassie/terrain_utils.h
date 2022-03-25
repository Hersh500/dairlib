//
// Created by hersh on 11/8/21.
//
#ifndef DAIRLIB_TERRAIN_UTILS_H
#define DAIRLIB_TERRAIN_UTILS_H
#include "drake/multibody/plant/multibody_plant.h"
#include "drake/geometry/scene_graph.h"

using drake::geometry::SceneGraph;
using drake::multibody::MultibodyPlant;

namespace dairlib {
    void generateRandomObstacles(MultibodyPlant<double> *plant,
                                 std::pair<double, double> x_lims,
                                 std::pair<double, double> y_lims,
                                 unsigned int num_obs);

    void generateRandomObstacles(SceneGraph<double>& graph,
                                 std::pair<double, double> x_lims,
                                 std::pair<double, double> y_lims);

    void generateRandomGaps(MultibodyPlant<double> *plant,
                            std::pair<double, double> gap_lims);

    void generateRandomSteps(MultibodyPlant<double> *plant,
                             std::pair<double, double> step_lims);

    void generateFixedSteps(MultibodyPlant<double> *plant);


    void generateReferenceTerrain(MultibodyPlant<double> *plant);

    void generateObstacleCourse(MultibodyPlant<double> *plant, double sigma_x, double sigma_y);

    void generateDitchesObstacleCourse(MultibodyPlant<double> *plant, double dropout_pct);

}

#endif //DAIRLIB_TERRAIN_UTILS_H
