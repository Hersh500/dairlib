#include "examples/Cassie/osc/high_level_const_speed_command.h"

#include <math.h>

#include <string>

#include "dairlib/lcmt_cassie_out.hpp"
#include "multibody/multibody_utils.h"

#include "drake/math/quaternion.h"

using std::cout;
using std::endl;
using std::string;

using Eigen::MatrixXd;
using Eigen::Vector2d;
using Eigen::Vector3d;
using Eigen::Vector4d;
using Eigen::VectorXd;

using Eigen::Quaterniond;

using dairlib::systems::OutputVector;

using drake::systems::BasicVector;
using drake::systems::Context;
using drake::systems::DiscreteUpdateEvent;
using drake::systems::DiscreteValues;
using drake::systems::EventStatus;
using drake::systems::LeafSystem;

using drake::multibody::JacobianWrtVariable;
using drake::trajectories::PiecewisePolynomial;

namespace dairlib {
namespace cassie {
namespace osc {

HighLevelConstSpeedCommand::HighLevelConstSpeedCommand(
    const drake::multibody::MultibodyPlant<double>& plant,
    drake::systems::Context<double>* context, double vel_scale_rot,
    double vel_scale_trans, int footstep_option)
    : HighLevelConstSpeedCommand(plant, context, footstep_option) {
  cassie_out_port_ =
      this->DeclareAbstractInputPort("lcmt_cassie_output",
                                     drake::Value<dairlib::lcmt_cassie_out>{})
          .get_index();
  use_radio_command_ = true;
  vel_scale_rot_ = vel_scale_rot;
  vel_scale_trans_ = vel_scale_trans;
}

HighLevelConstSpeedCommand::HighLevelConstSpeedCommand(
    const drake::multibody::MultibodyPlant<double>& plant,
    drake::systems::Context<double>* context,
    const Eigen::Vector2d& desired_vel, double ramp_time, int footstep_option)
    : HighLevelConstSpeedCommand(plant, context, footstep_option) {
  use_radio_command_ = false;
  desired_vel_ = desired_vel;
  ramp_time_ = ramp_time;
}

HighLevelConstSpeedCommand::HighLevelConstSpeedCommand(
    const drake::multibody::MultibodyPlant<double>& plant,
    drake::systems::Context<double>* context, int footstep_option)
    : plant_(plant),
      context_(context),
      world_(plant_.world_frame()),
      pelvis_(plant_.GetBodyByName("pelvis")) {
  DRAKE_DEMAND(0 <= footstep_option && footstep_option <= 1);

  state_port_ =
      this->DeclareVectorInputPort(OutputVector<double>(plant.num_positions(),
                                                        plant.num_velocities(),
                                                        plant.num_actuators()))
          .get_index();

  yaw_port_ = this->DeclareVectorOutputPort(
                      BasicVector<double>(1),
                      &HighLevelConstSpeedCommand::CopyHeadingAngle)
                  .get_index();
  xy_port_ = this->DeclareVectorOutputPort(
                     BasicVector<double>(2),
                     &HighLevelConstSpeedCommand::CopyDesiredHorizontalVel)
                 .get_index();
  // Declare update event
  DeclarePerStepDiscreteUpdateEvent(
      &HighLevelConstSpeedCommand::DiscreteVariableUpdate);

  // Discrete state which stores the desired yaw velocity
  des_vel_idx_ = DeclareDiscreteState(VectorXd::Zero(3));
  // Control gains
  if (footstep_option == 0) {
    kp_pos_sagital_ = 1.0;
    kd_pos_sagital_ = 0.2;

    kp_pos_lateral_ = 0.5;
    kd_pos_lateral_ = 0.1;
    vel_max_lateral_ = 0.5;
  } else if (footstep_option == 1) {
    kp_pos_sagital_ = 1.0;
    kd_pos_sagital_ = 1.0;

    kp_pos_lateral_ = 0.25;
    kd_pos_lateral_ = 1.0;
    vel_max_lateral_ = 0.8;
  }
}

EventStatus HighLevelConstSpeedCommand::DiscreteVariableUpdate(
    const Context<double>& context,
    DiscreteValues<double>* discrete_state) const {
  if (use_radio_command_) {
    const auto& cassie_out = this->EvalInputValue<dairlib::lcmt_cassie_out>(
        context, cassie_out_port_);  // TODO(yangwill) make sure there is a
    // message available
    // des_vel indices: 0: yaw_vel (right joystick left/right)
    //                  1: saggital_vel (left joystick up/down)
    //                  2: lateral_vel (left joystick left/right)
    Vector3d des_vel;
    des_vel << -1 * vel_scale_rot_ * cassie_out->pelvis.radio.channel[3],
        vel_scale_trans_ * cassie_out->pelvis.radio.channel[0],
        -1 * vel_scale_trans_ * cassie_out->pelvis.radio.channel[1];
    discrete_state->get_mutable_vector(des_vel_idx_).set_value(des_vel);
  } else {
    discrete_state->get_mutable_vector(des_vel_idx_)
        .set_value(CalcCommandFromTargetPosition(context));
  }

  return EventStatus::Succeeded();
}

VectorXd HighLevelConstSpeedCommand::CalcCommandFromTargetPosition(
    const Context<double>& context) const {
  // Read in current state
  const OutputVector<double>* robotOutput =
      (OutputVector<double>*)this->EvalVectorInput(context, state_port_);
  VectorXd q = robotOutput->GetPositions();
  VectorXd v = robotOutput->GetVelocities();

  plant_.SetPositions(context_, q);

  // Curren time
  double timestamp = robotOutput->get_timestamp();
  auto current_time = static_cast<double>(timestamp);

  //////////// Get desired yaw velocity ////////////
  // Get approximated heading angle of pelvis
  Vector3d pelvis_heading_vec =
      plant_.EvalBodyPoseInWorld(*context_, pelvis_).rotation().col(0);
  double approx_pelvis_yaw =
      atan2(pelvis_heading_vec(1), pelvis_heading_vec(0));

  // Get desired heading angle of pelvis
  double desired_yaw = atan2(0, 1);

  // Get current yaw velocity
  double yaw_vel = v(2);

  // PD position control
  double des_yaw_vel =
      kp_yaw_ * (desired_yaw - approx_pelvis_yaw) + kd_yaw_ * (-yaw_vel);
  des_yaw_vel = std::min(vel_max_yaw_, std::max(-vel_max_yaw_, des_yaw_vel));

  //////////// Get desired horizontal vel ////////////
  // Apply walking speed control only when the robot is facing the target
  // position.
  Eigen::Vector2d desired_vel;
  if (current_time < ramp_time_) {
    desired_vel = (current_time / ramp_time_) * desired_vel_;
  } else {
    desired_vel = desired_vel_;
  }

  Vector3d des_vel;
  des_vel << des_yaw_vel, desired_vel;

  return des_vel;
}

void HighLevelConstSpeedCommand::CopyHeadingAngle(
    const Context<double>& context, BasicVector<double>* output) const {
  double desired_heading_pos =
      context.get_discrete_state(des_vel_idx_).get_value()(0);
  // Assign
  output->get_mutable_value() << desired_heading_pos;
}

void HighLevelConstSpeedCommand::CopyDesiredHorizontalVel(
    const Context<double>& context, BasicVector<double>* output) const {
  auto delta_CP_3D_global =
      context.get_discrete_state(des_vel_idx_).get_value().tail(2);

  // Assign
  output->get_mutable_value() = delta_CP_3D_global;
}

}  // namespace osc
}  // namespace cassie
}  // namespace dairlib
