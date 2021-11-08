//
// Created by hersh on 11/8/21.
//
#ifndef DAIRLIB_TERRAIN_UTILS_H
#define DAIRLIB_TERRAIN_UTILS_H
#include "drake/geometry/scene_graph.h"

using drake::geometry::SceneGraph;

namespace dairlib {
    void generateRandomObstacles(SceneGraph<double> *graph, float );
}

#endif //DAIRLIB_TERRAIN_UTILS_H
