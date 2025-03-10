#include "multibody/geom_geom_collider.h"

#include "drake/math/rotation_matrix.h"

using Eigen::Vector3d;
using Eigen::Matrix;
using drake::EigenPtr;
using drake::MatrixX;
using drake::geometry::GeometryId;
using drake::geometry::SignedDistancePair;
using drake::multibody::MultibodyPlant;
using drake::systems::Context;

namespace dairlib {
namespace multibody {

template <typename T>
GeomGeomCollider<T>::GeomGeomCollider(
    const drake::multibody::MultibodyPlant<T>& plant,
    const drake::geometry::GeometryId geometry_id_A,
    const drake::geometry::GeometryId geometry_id_B,
    const int num_friction_directions)
    : plant_(plant),
      geometry_id_A_(geometry_id_A),
      geometry_id_B_(geometry_id_B),
      num_friction_directions_(num_friction_directions) ,
      planar_normal_(Vector3d::Zero()) {
  if (num_friction_directions == 1) {
    throw std::runtime_error(
      "GeomGeomCollider cannot specificy 1 friction direction unless "
          "using planar constructor");
  }
}

template <typename T>
GeomGeomCollider<T>::GeomGeomCollider(
    const drake::multibody::MultibodyPlant<T>& plant,
    const drake::geometry::GeometryId geometry_id_A,
    const drake::geometry::GeometryId geometry_id_B,
    const Eigen::Vector3d planar_normal)
    : plant_(plant),
      geometry_id_A_(geometry_id_A),
      geometry_id_B_(geometry_id_B),
      num_friction_directions_(1),
      planar_normal_(planar_normal) {}

template <typename T>
T GeomGeomCollider<T>::Eval(const Context<T>& context,
                             EigenPtr<MatrixX<T>> J) {

  const auto& query_port = plant_.get_geometry_query_input_port();
  const auto& query_object =
      query_port.template Eval<drake::geometry::QueryObject<T>>(context);

  const SignedDistancePair<T> signed_distance_pair =
      query_object.ComputeSignedDistancePairClosestPoints(geometry_id_A_,
                                                          geometry_id_B_);

  const auto& inspector = query_object.inspector();
  const auto frame_A_id = inspector.GetFrameId(geometry_id_A_);
  const auto frame_B_id = inspector.GetFrameId(geometry_id_B_);
  const auto& frameA = plant_.GetBodyFromFrameId(frame_A_id)->body_frame();
  const auto& frameB = plant_.GetBodyFromFrameId(frame_B_id)->body_frame();

  const Eigen::Vector3d& p_ACa =
      inspector.GetPoseInFrame(geometry_id_A_).template cast<T>() *
          signed_distance_pair.p_ACa;

  Matrix<double, 3, Eigen::Dynamic> Jq_v_BCa_W(3, plant_.num_positions());
  auto wrt = drake::multibody::JacobianWrtVariable::kQDot;

  plant_.CalcJacobianTranslationalVelocity(context, wrt,
                                            frameA, p_ACa, frameB,
                                            plant_.world_frame(), &Jq_v_BCa_W);

  // Compute force basis
  Matrix<double, 3, Eigen::Dynamic> force_basis(
      3, 2 * num_friction_directions_ + 1);
  force_basis.col(0) = signed_distance_pair.nhat_BA_W;

  if (num_friction_directions_ == 1) {
    // Then we have a planar problem
    force_basis.col(1) =
        signed_distance_pair.nhat_BA_W.cross(planar_normal_);
    force_basis.col(1).normalize();
    force_basis.col(2) = -force_basis.col(1);
  } else {
    // Build friction basis
    const auto& R_WC =
        drake::math::RotationMatrix<T>::MakeFromOneVector(
            signed_distance_pair.nhat_BA_W, 0);
    for (int i = 0; i < num_friction_directions_; i++) {
      double theta = (M_PI * i) / num_friction_directions_;
      force_basis.col(2*i + 1) = R_WC * Vector3d(0, cos(theta), sin(theta));
      force_basis.col(2*i + 2) = -force_basis.col(2*i + 1);
    }
  }

  *J = force_basis.transpose() * Jq_v_BCa_W;

  return signed_distance_pair.distance;

}

}  // namespace multibody
}  // namespace dairlib


template class dairlib::multibody::GeomGeomCollider<double>;