//
// Created by hersh on 11/8/21.
//
#include <drake/geometry/shape_specification.h>
#include "examples/Cassie/terrain_utils.h"
#include "drake/geometry/scene_graph.h"

using drake::geometry::SceneGraph;
using drake::geometry::SourceId;
using drake::geometry::GeometryInstance;
using std::make_unique;

namespace dairlib {
    void generateRandomObstacles(SceneGraph<double> *scene_graph) {
        SourceId terrain_geom_id = scene_graph->RegisterSource("terrain_adder");
        for (int i = 0; i < 10; i++) {
            // Generate a random pose for this object; maybe based on some stochastic process?
            scene_graph->RegisterAnchoredGeometry(
                    terrain_geom_id,std::make_unique<GeometryInstance>(GeometryInstance(pose, make_unique<Box>(0.5), "collision")));

        }

    }

    // Read a PNG heightmap and add boxes to simulate this.
    // Does this slow down the simulator a lot?
    void generateObstaclesFromHeightmap(SceneGraph<double> *scene_graph,
                                        float pixels_to_m, float val_to_m,
                                        const std::string *heightmap_name) {

    }
}

