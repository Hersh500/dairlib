#include <pybind11/eigen.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "multibody/multibody_utils.h"
#include "multibody/multipose_visualizer.h"

namespace py = pybind11;

namespace dairlib {
namespace pydairlib {

using multibody::MultiposeVisualizer;

PYBIND11_MODULE(multibody, m) {
  py::module::import("pydrake.all");

  m.doc() = "Binding utility functions for MultibodyPlant";

  py::class_<MultiposeVisualizer>(m, "MultiposeVisualizer")
      .def(py::init<std::string, int, std::string>())
      .def(py::init<std::string, int, double, std::string>())
      .def(py::init<std::string, int, Eigen::VectorXd, std::string>())
      .def("DrawPoses", &MultiposeVisualizer::DrawPoses, py::arg("poses"));

  m.def("makeNameToPositionsMap",
        &dairlib::multibody::makeNameToPositionsMap<double>, py::arg("plant"))
      .def("makeNameToVelocitiesMap",
           &dairlib::multibody::makeNameToVelocitiesMap<double>,
           py::arg("plant"))
      .def("makeNameToActuatorsMap",
           &dairlib::multibody::makeNameToActuatorsMap<double>,
           py::arg("plant"))
      .def("createStateNameVectorFromMap",
           &dairlib::multibody::createStateNameVectorFromMap<double>,
           py::arg("plant"))
      .def("createActuatorNameVectorFromMap",
           &dairlib::multibody::createActuatorNameVectorFromMap<double>,
           py::arg("plant"))
      .def("addFlatTerrain", &dairlib::multibody::addFlatTerrain<double>,
           py::arg("plant"), py::arg("scene_graph"), py::arg("mu_static"),
           py::arg("mu_kinetic"),
           py::arg("normal_W") = Eigen::Vector3d(0, 0, 1),
           py::arg("show_ground") = 1);
}

}  // namespace pydairlib
}  // namespace dairlib
