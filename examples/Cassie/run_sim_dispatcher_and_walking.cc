#include <chrono>
#include <memory>
#include <gflags/gflags.h>

#include "dairlib/lcmt_cassie_out.hpp"
#include "dairlib/lcmt_robot_input.hpp"
#include "dairlib/lcmt_robot_output.hpp"
#include "examples/Cassie/cassie_fixed_point_solver.h"
#include "examples/Cassie/cassie_state_estimator.h"
#include "examples/Cassie/cassie_utils.h"
#include "examples/Cassie/networking/cassie_output_receiver.h"
#include "examples/Cassie/networking/cassie_output_sender.h"
#include "examples/Cassie/networking/simple_cassie_udp_subscriber.h"
#include "examples/Cassie/osc/heading_traj_generator.h"
#include "examples/Cassie/osc/high_level_command.h"
#include "examples/Cassie/osc/walking_speed_control.h"
#include "examples/Cassie/simulator_drift.h"
#include "multibody/kinematic/fixed_joint_evaluator.h"
#include "multibody/kinematic/kinematic_evaluator_set.h"
#include "multibody/kinematic/world_point_evaluator.h"
#include "multibody/multibody_solvers.h"
#include "multibody/multibody_utils.h"
#include "systems/controllers/fsm_event_time.h"
#include "systems/controllers/lipm_traj_gen.h"
#include "systems/controllers/osc/operational_space_control.h"
#include "systems/controllers/swing_ft_traj_gen.h"
#include "systems/controllers/time_based_fsm.h"
#include "systems/framework/lcm_driven_loop.h"
#include "systems/framework/output_vector.h"
#include "systems/primitives/subvector_pass_through.h"
#include "systems/robot_lcm_systems.h"

#include "drake/common/yaml/yaml_read_archive.h"
#include "drake/geometry/geometry_visualization.h"
#include "drake/lcm/drake_lcm.h"
#include "drake/lcmt_contact_results_for_viz.hpp"
#include "drake/multibody/plant/contact_results_to_lcm.h"
#include "drake/solvers/choose_best_solver.h"
#include "drake/solvers/snopt_solver.h"
#include "drake/solvers/solve.h"
#include "drake/systems/analysis/runge_kutta2_integrator.h"
#include "drake/systems/analysis/simulator.h"
#include "drake/systems/framework/diagram.h"
#include "drake/systems/framework/diagram_builder.h"
#include "drake/systems/lcm/lcm_interface_system.h"
#include "drake/systems/lcm/lcm_publisher_system.h"
#include "drake/systems/lcm/lcm_subscriber_system.h"
#include "drake/systems/primitives/discrete_time_delay.h"
#include "drake/systems/primitives/zero_order_hold.h"

namespace dairlib {

using std::cout;
using std::endl;
using std::vector;

using Eigen::Matrix3d;
using Eigen::MatrixXd;
using Eigen::Vector3d;
using Eigen::VectorXd;

using drake::geometry::SceneGraph;
using drake::math::RotationMatrix;
using drake::multibody::ContactResultsToLcmSystem;
using drake::multibody::Frame;
using drake::multibody::MultibodyPlant;
using drake::systems::Context;
using drake::systems::DiagramBuilder;
using drake::systems::Simulator;
using drake::systems::TriggerType;
using drake::systems::lcm::LcmPublisherSystem;
using drake::systems::lcm::LcmSubscriberSystem;
using drake::systems::lcm::TriggerTypeSet;

using dairlib::systems::SubvectorPassThrough;

using multibody::FixedJointEvaluator;

using systems::controllers::ComTrackingData;
using systems::controllers::JointSpaceTrackingData;
using systems::controllers::RotTaskSpaceTrackingData;
using systems::controllers::TransTaskSpaceTrackingData;

// Controller
DEFINE_double(drift_rate, 0.0, "Drift rate for floating-base state");
DEFINE_string(channel_x, "CASSIE_STATE_SIMULATION",
              "LCM channel for receiving state. "
              "Use CASSIE_STATE_SIMULATION to get state from simulator, and "
              "use CASSIE_STATE_DISPATCHER to get state from state estimator");
DEFINE_string(channel_u, "CASSIE_INPUT",
              "The name of the channel which publishes command");
DEFINE_bool(use_radio, false,
            "Set to true if sending high level commands from radio controller");
DEFINE_string(
    cassie_out_channel, "CASSIE_OUTPUT_ECHO",
    "The name of the channel to receive the cassie out structure from.");
DEFINE_string(gains_filename, "examples/Cassie/osc/osc_walking_gains.yaml",
              "Filepath containing gains");
DEFINE_bool(publish_osc_data, true,
            "whether to publish lcm messages for OscTrackData");
DEFINE_bool(print_osc, false, "whether to print the osc debug message or not");

DEFINE_bool(is_two_phase, false,
            "true: only right/left single support"
            "false: both double and single support");
DEFINE_int32(
    footstep_option, 1,
    "0 uses the capture point\n"
    "1 uses the neutral point derived from LIPM given the stance duration");

// Simulation parameters.
DEFINE_double(target_realtime_rate, 1.0,
              "Desired rate relative to real time.  See documentation for "
              "Simulator::set_target_realtime_rate() for details.");
DEFINE_bool(time_stepping, true,
            "If 'true', the plant is modeled as a "
            "discrete system with periodic updates. "
            "If 'false', the plant is modeled as a continuous system.");
DEFINE_double(dt, 8e-5,
              "The step size to use for time_stepping, ignored for continuous");
DEFINE_double(v_stiction, 1e-3, "Stiction tolernace (m/s)");
DEFINE_double(penetration_allowance, 1e-5,
              "Penetration allowance for the contact model. Nearly equivalent"
              " to (m)");
DEFINE_double(end_time, std::numeric_limits<double>::infinity(),
              "End time for simulator");
DEFINE_double(publish_rate, 1000, "Publish rate for simulator");
DEFINE_double(init_height, 1.0,
              "Initial starting height of the pelvis above "
              "ground");

// Dispatcher parameters
DEFINE_bool(print_ekf_info, false, "Print ekf information to the terminal");
// Testing mode
DEFINE_int64(test_mode, -1,
             "-1: Regular EKF (not testing mode). "
             "0: both feet always in contact with ground. "
             "1: both feet never in contact with ground. "
             "2: both feet always in contact with the ground until contact is"
             " detected in which case it swtiches to test mode -1.");
//
DEFINE_bool(use_dispatcher, true, "Use estimated state in the controller");

// Learning parameters
DEFINE_double(w_accel, 0.000001, "");
DEFINE_double(w_soft_constraint, 8000, "");
DEFINE_double(w_swing_toe, 100, "");
DEFINE_double(swing_toe_kp, 20, "");
DEFINE_double(swing_toe_kd, 15, "");
DEFINE_double(w_hip_yaw, 100, "");
DEFINE_double(hip_yaw_kp, 100, "");
DEFINE_double(hip_yaw_kd, 10, "");
DEFINE_double(w_com_z, 10, "");
DEFINE_double(k_p_com_z, 400, "");
DEFINE_double(k_d_com_z, 400, "");
DEFINE_double(w_pelvis_balance_x, 400, "");
DEFINE_double(w_pelvis_balance_y, 200, "");
DEFINE_double(k_p_pelvis_balance_x, 400, "");
DEFINE_double(k_p_pelvis_balance_y, 400, "");
DEFINE_double(k_d_pelvis_balance_x, 10, "");
DEFINE_double(k_d_pelvis_balance_y, 10, "");
DEFINE_double(w_pelvis_heading_z, 10, "");
DEFINE_double(k_p_pelvis_heading_z, 10, "");
DEFINE_double(k_d_pelvis_heading_z, 10, "");
DEFINE_double(w_swing_foot_x, 400, "");
DEFINE_double(w_swing_foot_y, 400, "");
DEFINE_double(w_swing_foot_z, 400, "");
DEFINE_double(k_p_swing_foot_x, 200, "");
DEFINE_double(k_p_swing_foot_y, 400, "");
DEFINE_double(k_p_swing_foot_z, 400, "");
DEFINE_double(k_d_swing_foot_x, 10, "");
DEFINE_double(k_d_swing_foot_y, 10, "");
DEFINE_double(k_d_swing_foot_z, 10, "");

DEFINE_double(mid_foot_height, 0.05, "");

DEFINE_double(double_support_duration, 0.02, "");

DEFINE_double(knee_spring_left, 1500, "");
DEFINE_double(knee_spring_right, 1500, "");
DEFINE_double(ankle_spring_left, 1250, "");
DEFINE_double(ankle_spring_right, 1250, "");

DEFINE_double(pelvis_disturbnace_xdot, 0, "in m/s");
DEFINE_double(pelvis_disturbnace_ydot, 0, "in m/s");
DEFINE_double(pelvis_disturbnace_zdot, 0, "in m/s");

DEFINE_double(random_joint_damping_min, -1, "");
DEFINE_double(random_joint_damping_max, -1, "");

DEFINE_string(sample_id, "", "");
DEFINE_bool(print_gains, false, "");
DEFINE_bool(pub_state_from_dispatcher, false, "");

struct OSCWalkingGains {
  int rows;
  int cols;
  double w_accel;
  double w_soft_constraint;
  std::vector<double> CoMW;
  std::vector<double> CoMKp;
  std::vector<double> CoMKd;
  std::vector<double> PelvisHeadingW;
  std::vector<double> PelvisHeadingKp;
  std::vector<double> PelvisHeadingKd;
  std::vector<double> PelvisBalanceW;
  std::vector<double> PelvisBalanceKp;
  std::vector<double> PelvisBalanceKd;
  std::vector<double> SwingFootW;
  std::vector<double> SwingFootKp;
  std::vector<double> SwingFootKd;
  double w_swing_toe;
  double swing_toe_kp;
  double swing_toe_kd;
  double w_hip_yaw;
  double hip_yaw_kp;
  double hip_yaw_kd;
  double center_line_offset;
  double footstep_offset;
  double mid_foot_height;
  double final_foot_height;
  double final_foot_velocity_z;
  double lipm_height;
  double double_support_duration;

  template <typename Archive>
  void Serialize(Archive* a) {
    a->Visit(DRAKE_NVP(rows));
    a->Visit(DRAKE_NVP(cols));
    a->Visit(DRAKE_NVP(w_accel));
    a->Visit(DRAKE_NVP(w_soft_constraint));
    a->Visit(DRAKE_NVP(CoMW));
    a->Visit(DRAKE_NVP(CoMKp));
    a->Visit(DRAKE_NVP(CoMKd));
    a->Visit(DRAKE_NVP(PelvisHeadingW));
    a->Visit(DRAKE_NVP(PelvisHeadingKp));
    a->Visit(DRAKE_NVP(PelvisHeadingKd));
    a->Visit(DRAKE_NVP(PelvisBalanceW));
    a->Visit(DRAKE_NVP(PelvisBalanceKp));
    a->Visit(DRAKE_NVP(PelvisBalanceKd));
    a->Visit(DRAKE_NVP(SwingFootW));
    a->Visit(DRAKE_NVP(SwingFootKp));
    a->Visit(DRAKE_NVP(SwingFootKd));
    a->Visit(DRAKE_NVP(w_swing_toe));
    a->Visit(DRAKE_NVP(swing_toe_kp));
    a->Visit(DRAKE_NVP(swing_toe_kd));
    a->Visit(DRAKE_NVP(w_hip_yaw));
    a->Visit(DRAKE_NVP(hip_yaw_kp));
    a->Visit(DRAKE_NVP(hip_yaw_kd));
    // swing foot heuristics
    a->Visit(DRAKE_NVP(mid_foot_height));
    a->Visit(DRAKE_NVP(center_line_offset));
    a->Visit(DRAKE_NVP(footstep_offset));
    a->Visit(DRAKE_NVP(final_foot_height));
    a->Visit(DRAKE_NVP(final_foot_velocity_z));
    // lipm heursitics
    a->Visit(DRAKE_NVP(lipm_height));

    a->Visit(DRAKE_NVP(double_support_duration));
  }
};

// set the initial state for the EKF.
void setInitialEkfState(double t0, const VectorXd& q_init,
                        const drake::systems::Diagram<double>& diagram,
                        const systems::CassieStateEstimator& state_estimator,
                        drake::systems::Context<double>* diagram_context) {
  // Set initial time and floating base position
  auto& state_estimator_context =
      diagram.GetMutableSubsystemContext(state_estimator, diagram_context);
  state_estimator.setPreviousTime(&state_estimator_context, t0);
  state_estimator.setInitialPelvisPose(&state_estimator_context, q_init.head(4),
                                       q_init.segment<3>(4));
  // Set initial imu value
  // Note that initial imu values are all 0 if the robot is dropped from the air
  Eigen::VectorXd init_prev_imu_value = Eigen::VectorXd::Zero(6);
  init_prev_imu_value << 0, 0, 0, 0, 0, 9.81;
  state_estimator.setPreviousImuMeasurement(&state_estimator_context,
                                            init_prev_imu_value);
}

int DoMain(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  std::string suffix = FLAGS_sample_id;
  //  std::string suffix = "";
  //  if (FLAGS_sample_idx >= 0) {
  //    suffix = std::to_string(FLAGS_sample_idx);
  //  }

  ////// Simulator //////

  std::string urdf = "examples/Cassie/urdf/cassie_v2.urdf";
  // urdf = "examples/Cassie/urdf/cassie_fixed_springs.urdf";

  drake::logging::set_log_level("err");  // ignore warnings about joint limits

  std::vector<double> random_joint_damping_range = {};
  if (FLAGS_random_joint_damping_min >= 0 and
      FLAGS_random_joint_damping_max >= 0) {
    random_joint_damping_range = {FLAGS_random_joint_damping_min,
                                  FLAGS_random_joint_damping_max};
  }

  // Plant/System initialization
  DiagramBuilder<double> builder;
  SceneGraph<double>& scene_graph = *builder.AddSystem<SceneGraph>();
  scene_graph.set_name("scene_graph");

  const double time_step = FLAGS_time_stepping ? FLAGS_dt : 0.0;
  MultibodyPlant<double>& plant_sim =
      *builder.AddSystem<MultibodyPlant>(time_step);
  multibody::addFlatTerrain(&plant_sim, &scene_graph, .8, .8);
  addCassieMultibody(&plant_sim, &scene_graph, true, urdf, true, true,
                     {FLAGS_knee_spring_left, FLAGS_knee_spring_right,
                      FLAGS_ankle_spring_left, FLAGS_ankle_spring_right},
                     random_joint_damping_range);
  plant_sim.Finalize();

  plant_sim.set_penetration_allowance(FLAGS_penetration_allowance);
  plant_sim.set_stiction_tolerance(FLAGS_v_stiction);

  // Create lcm systems.
  drake::lcm::DrakeLcm lcm_local("udpm://239.255.76.67:7667?ttl=0");
  //  auto input_sub =
  //      builder.AddSystem(LcmSubscriberSystem::Make<dairlib::lcmt_robot_input>(
  //          "CASSIE_INPUT", &lcm_local));
  auto input_receiver =
      builder.AddSystem<systems::RobotInputReceiver>(plant_sim);
  auto passthrough = builder.AddSystem<SubvectorPassThrough>(
      input_receiver->get_output_port(0).size(), 0,
      plant_sim.get_actuation_input_port().size());
  auto state_pub =
      builder.AddSystem(LcmPublisherSystem::Make<dairlib::lcmt_robot_output>(
          "CASSIE_STATE_SIMULATION" + suffix, &lcm_local,
          1.0 / FLAGS_publish_rate));
  auto state_sender = builder.AddSystem<systems::RobotOutputSender>(plant_sim);

  // Contact Information
  ContactResultsToLcmSystem<double>& contact_viz =
      *builder.template AddSystem<ContactResultsToLcmSystem<double>>(plant_sim);
  contact_viz.set_name("contact_visualization");
  //  auto& contact_results_publisher = *builder.AddSystem(
  //      LcmPublisherSystem::Make<drake::lcmt_contact_results_for_viz>(
  //          "CASSIE_CONTACT_DRAKE" + suffix, &lcm_local, 1.0 /
  //          FLAGS_publish_rate));
  //  contact_results_publisher.set_name("contact_results_publisher");

  // Sensor aggregator and publisher of lcmt_cassie_out
  const auto& sensor_aggregator =
      AddImuAndAggregator(&builder, plant_sim, passthrough->get_output_port());
  //  auto sensor_pub =
  //      builder.AddSystem(LcmPublisherSystem::Make<dairlib::lcmt_cassie_out>(
  //          "CASSIE_OUTPUT" + suffix, &lcm_local, 1.0 / FLAGS_publish_rate));

  // Connect leaf systems
  //  builder.Connect(*input_sub, *input_receiver);
  builder.Connect(*input_receiver, *passthrough);
  builder.Connect(plant_sim.get_state_output_port(),
                  state_sender->get_input_port_state());
  builder.Connect(*state_sender, *state_pub);
  builder.Connect(
      plant_sim.get_geometry_poses_output_port(),
      scene_graph.get_source_pose_port(plant_sim.get_source_id().value()));
  builder.Connect(scene_graph.get_query_output_port(),
                  plant_sim.get_geometry_query_input_port());
  builder.Connect(plant_sim.get_contact_results_output_port(),
                  contact_viz.get_input_port(0));
  //  builder.Connect(contact_viz.get_output_port(0),
  //                  contact_results_publisher.get_input_port());
  //  builder.Connect(sensor_aggregator.get_output_port(0),
  //                  sensor_pub->get_input_port());

  // Zero order hold for the robot input
  //  builder.Connect(passthrough->get_output_port(),
  //                  plant_sim.get_actuation_input_port());
  double kPeriod = 0.002;
  auto input_zero_order_hold =
      builder.AddSystem<drake::systems::ZeroOrderHold<double>>(
          kPeriod, plant_sim.num_actuators());
  builder.Connect(passthrough->get_output_port(),
                  input_zero_order_hold->get_input_port());
  bool is_time_delay = true;
  double time_delay = 0.014;
  if (is_time_delay) {
    auto time_delay_block =
        builder.AddSystem<drake::systems::DiscreteTimeDelay<double>>(
            kPeriod, int(time_delay / kPeriod), plant_sim.num_actuators());
    builder.Connect(input_zero_order_hold->get_output_port(),
                    time_delay_block->get_input_port());
    builder.Connect(time_delay_block->get_output_port(),
                    plant_sim.get_actuation_input_port());

  } else {
    builder.Connect(input_zero_order_hold->get_output_port(),
                    plant_sim.get_actuation_input_port());
  }

  ////// Dispatcher //////

  // Build Cassie MBP
  drake::multibody::MultibodyPlant<double> plant_ctrl(0.0);
  addCassieMultibody(&plant_ctrl, nullptr, true /*floating base*/, urdf,
                     true /*spring model*/, false /*loop closure*/);
  plant_ctrl.Finalize();

  // Evaluators for fourbar linkages
  multibody::KinematicEvaluatorSet<double> fourbar_evaluator(plant_ctrl);
  auto left_loop = LeftLoopClosureEvaluator(plant_ctrl);
  auto right_loop = RightLoopClosureEvaluator(plant_ctrl);
  fourbar_evaluator.add_evaluator(&left_loop);
  fourbar_evaluator.add_evaluator(&right_loop);
  // Evaluators for contact points (The position doesn't matter. It's not used
  // in OSC)
  multibody::KinematicEvaluatorSet<double> left_contact_evaluator(plant_ctrl);
  auto left_toe = LeftToeFront(plant_ctrl);
  auto left_heel = LeftToeRear(plant_ctrl);
  auto left_toe_evaluator = multibody::WorldPointEvaluator(
      plant_ctrl, left_toe.first, left_toe.second, Matrix3d::Identity(),
      Vector3d::Zero(), {1, 2});
  auto left_heel_evaluator = multibody::WorldPointEvaluator(
      plant_ctrl, left_heel.first, left_heel.second, Matrix3d::Identity(),
      Vector3d::Zero(), {0, 1, 2});
  left_contact_evaluator.add_evaluator(&left_toe_evaluator);
  left_contact_evaluator.add_evaluator(&left_heel_evaluator);
  multibody::KinematicEvaluatorSet<double> right_contact_evaluator(plant_ctrl);
  auto right_toe = RightToeFront(plant_ctrl);
  auto right_heel = RightToeRear(plant_ctrl);
  auto right_toe_evaluator = multibody::WorldPointEvaluator(
      plant_ctrl, right_toe.first, right_toe.second, Matrix3d::Identity(),
      Vector3d::Zero(), {1, 2});
  auto right_heel_evaluator = multibody::WorldPointEvaluator(
      plant_ctrl, right_heel.first, right_heel.second, Matrix3d::Identity(),
      Vector3d::Zero(), {0, 1, 2});
  right_contact_evaluator.add_evaluator(&right_toe_evaluator);
  right_contact_evaluator.add_evaluator(&right_heel_evaluator);

  // Create state estimator
  auto state_estimator = builder.AddSystem<systems::CassieStateEstimator>(
      plant_ctrl, &fourbar_evaluator, &left_contact_evaluator,
      &right_contact_evaluator, false, FLAGS_print_ekf_info, FLAGS_test_mode);

  // Connect appropriate input receiver for simulation
  systems::CassieOutputReceiver* cassie_output_receiver = nullptr;
  cassie_output_receiver = builder.AddSystem<systems::CassieOutputReceiver>();
  if (FLAGS_use_dispatcher) {
    builder.Connect(sensor_aggregator.get_output_port(0),
                    cassie_output_receiver->get_input_port(0));
  }
  builder.Connect(cassie_output_receiver->get_output_port(0),
                  state_estimator->get_input_port(0));

  // Create and connect RobotOutput publisher.
  auto robot_output_sender =
      builder.AddSystem<systems::RobotOutputSender>(plant_ctrl, true);

  //  // Create and connect contact estimation publisher.
  //  auto contact_pub =
  //      builder.AddSystem(LcmPublisherSystem::Make<dairlib::lcmt_contact>(
  //          "CASSIE_CONTACT_DISPATCHER", &lcm_local, 1.0 /
  //          FLAGS_publish_rate));
  //  builder.Connect(state_estimator->get_contact_output_port(),
  //                  contact_pub->get_input_port());
  //  // Create and connect contact estimation publisher.
  //  auto filtered_contact_pub =
  //      builder.AddSystem(LcmPublisherSystem::Make<dairlib::lcmt_contact>(
  //          "CASSIE_FILTERED_CONTACT_DISPATCHER", &lcm_local,
  //          1.0 / FLAGS_publish_rate));
  //  auto gm_contact_pub = builder.AddSystem(
  //      LcmPublisherSystem::Make<drake::lcmt_contact_results_for_viz>(
  //          "CASSIE_GM_CONTACT_DISPATCHER", &lcm_local, 1.0 /
  //          FLAGS_publish_rate));
  //  builder.Connect(state_estimator->get_filtered_contact_output_port(),
  //                  filtered_contact_pub->get_input_port());
  //  builder.Connect(state_estimator->get_gm_contact_output_port(),
  //                  gm_contact_pub->get_input_port());

  // Pass through to drop all but positions and velocities
  auto state_passthrough = builder.AddSystem<systems::SubvectorPassThrough>(
      state_estimator->get_robot_output_port().size(), 0,
      robot_output_sender->get_input_port_state().size());

  // Passthrough to pass efforts
  auto effort_passthrough = builder.AddSystem<systems::SubvectorPassThrough>(
      state_estimator->get_robot_output_port().size(),
      robot_output_sender->get_input_port_state().size(),
      robot_output_sender->get_input_port_effort().size());

  builder.Connect(state_estimator->get_robot_output_port(),
                  state_passthrough->get_input_port());
  builder.Connect(state_passthrough->get_output_port(),
                  robot_output_sender->get_input_port_state());

  builder.Connect(state_estimator->get_robot_output_port(),
                  effort_passthrough->get_input_port());
  builder.Connect(effort_passthrough->get_output_port(),
                  robot_output_sender->get_input_port_effort());

  if (FLAGS_pub_state_from_dispatcher) {
    auto state_pub_from_dispatcher =
        builder.AddSystem(LcmPublisherSystem::Make<dairlib::lcmt_robot_output>(
            "CASSIE_STATE_DISPATCHER" + suffix, &lcm_local,
            1.0 / FLAGS_publish_rate));
    builder.Connect(*robot_output_sender, *state_pub_from_dispatcher);
  }

  ////// Controller //////

  // Build Cassie MBP context
  auto context_w_spr = plant_ctrl.CreateDefaultContext();

  // Build the controller diagram

  OSCWalkingGains gains;
  const YAML::Node& root =
      YAML::LoadFile(FindResourceOrThrow(FLAGS_gains_filename));
  drake::yaml::YamlReadArchive(root).Accept(&gains);

  MatrixXd W_com = Eigen::Map<
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(
      gains.CoMW.data(), gains.rows, gains.cols);
  MatrixXd K_p_com = Eigen::Map<
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(
      gains.CoMKp.data(), gains.rows, gains.cols);
  MatrixXd K_d_com = Eigen::Map<
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(
      gains.CoMKd.data(), gains.rows, gains.cols);
  MatrixXd W_pelvis_heading = Eigen::Map<
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(
      gains.PelvisHeadingW.data(), gains.rows, gains.cols);
  MatrixXd K_p_pelvis_heading = Eigen::Map<
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(
      gains.PelvisHeadingKp.data(), gains.rows, gains.cols);
  MatrixXd K_d_pelvis_heading = Eigen::Map<
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(
      gains.PelvisHeadingKd.data(), gains.rows, gains.cols);
  MatrixXd W_pelvis_balance = Eigen::Map<
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(
      gains.PelvisBalanceW.data(), gains.rows, gains.cols);
  MatrixXd K_p_pelvis_balance = Eigen::Map<
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(
      gains.PelvisBalanceKp.data(), gains.rows, gains.cols);
  MatrixXd K_d_pelvis_balance = Eigen::Map<
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(
      gains.PelvisBalanceKd.data(), gains.rows, gains.cols);
  MatrixXd W_swing_foot = Eigen::Map<
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(
      gains.SwingFootW.data(), gains.rows, gains.cols);
  MatrixXd K_p_swing_foot = Eigen::Map<
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(
      gains.SwingFootKp.data(), gains.rows, gains.cols);
  MatrixXd K_d_swing_foot = Eigen::Map<
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(
      gains.SwingFootKd.data(), gains.rows, gains.cols);

  // Overwrite gains from flags
  gains.w_accel = FLAGS_w_accel;
  gains.w_soft_constraint = FLAGS_w_soft_constraint;
  gains.w_swing_toe = FLAGS_w_swing_toe;
  gains.swing_toe_kp = FLAGS_swing_toe_kp;
  gains.swing_toe_kd = FLAGS_swing_toe_kd;
  gains.w_hip_yaw = FLAGS_w_hip_yaw;
  gains.hip_yaw_kp = FLAGS_hip_yaw_kp;
  gains.hip_yaw_kd = FLAGS_hip_yaw_kd;
  gains.mid_foot_height = FLAGS_mid_foot_height;
  gains.double_support_duration = FLAGS_double_support_duration;
  W_com(2, 2) = FLAGS_w_com_z;
  K_p_com(2, 2) = FLAGS_k_p_com_z;
  K_d_com(2, 2) = FLAGS_k_d_com_z;
  W_pelvis_balance(0, 0) = FLAGS_w_pelvis_balance_x;
  W_pelvis_balance(1, 1) = FLAGS_w_pelvis_balance_y;
  K_p_pelvis_balance(0, 0) = FLAGS_k_p_pelvis_balance_x;
  K_p_pelvis_balance(1, 1) = FLAGS_k_p_pelvis_balance_y;
  K_d_pelvis_balance(0, 0) = FLAGS_k_d_pelvis_balance_x;
  K_d_pelvis_balance(1, 1) = FLAGS_k_d_pelvis_balance_y;
  W_pelvis_heading(2, 2) = FLAGS_w_pelvis_heading_z;
  K_p_pelvis_heading(2, 2) = FLAGS_k_p_pelvis_heading_z;
  K_d_pelvis_heading(2, 2) = FLAGS_k_d_pelvis_heading_z;
  W_swing_foot(0, 0) = FLAGS_w_swing_foot_x;
  W_swing_foot(1, 1) = FLAGS_w_swing_foot_y;
  W_swing_foot(2, 2) = FLAGS_w_swing_foot_z;
  K_p_swing_foot(0, 0) = FLAGS_k_p_swing_foot_x;
  K_p_swing_foot(1, 1) = FLAGS_k_p_swing_foot_y;
  K_p_swing_foot(2, 2) = FLAGS_k_p_swing_foot_z;
  K_d_swing_foot(0, 0) = FLAGS_k_d_swing_foot_x;
  K_d_swing_foot(1, 1) = FLAGS_k_d_swing_foot_y;
  K_d_swing_foot(2, 2) = FLAGS_k_d_swing_foot_z;

  if (FLAGS_print_gains) {
    auto clock_now = std::chrono::system_clock::now();
    std::time_t current_time = std::chrono::system_clock::to_time_t(clock_now);
    cout << "\nCurrent time: " << std::ctime(&current_time);
    std::cout << "w accel: " << gains.w_accel << std::endl;
    std::cout << "w soft constraint: " << gains.w_soft_constraint << std::endl;
    std::cout << "w_swing_toe: " << gains.w_swing_toe << std::endl;
    std::cout << "swing_toe_kp: " << gains.swing_toe_kp << std::endl;
    std::cout << "swing_toe_kd: " << gains.swing_toe_kd << std::endl;
    std::cout << "w_hip_yaw: " << gains.w_hip_yaw << std::endl;
    std::cout << "hip_yaw_kp: " << gains.hip_yaw_kp << std::endl;
    std::cout << "hip_yaw_kd: " << gains.hip_yaw_kd << std::endl;
    std::cout << "mid_foot_height: " << gains.mid_foot_height << std::endl;
    std::cout << "double_support_duration: \n"
              << gains.double_support_duration << std::endl;
    std::cout << "COM W: \n" << W_com << std::endl;
    std::cout << "COM Kp: \n" << K_p_com << std::endl;
    std::cout << "COM Kd: \n" << K_d_com << std::endl;
    std::cout << "Pelvis Balance W: \n" << W_pelvis_balance << std::endl;
    std::cout << "Pelvis Balance Kp: \n" << K_p_pelvis_balance << std::endl;
    std::cout << "Pelvis Balance Kd: \n" << K_d_pelvis_balance << std::endl;
    std::cout << "Pelvis Heading W: \n" << W_pelvis_heading << std::endl;
    std::cout << "Pelvis Heading Kp: \n" << K_p_pelvis_heading << std::endl;
    std::cout << "Pelvis Heading Kd: \n" << K_d_pelvis_heading << std::endl;
    std::cout << "Swing Foot W: \n" << W_swing_foot << std::endl;
    std::cout << "Swing Foot Kp: \n" << K_p_swing_foot << std::endl;
    std::cout << "Swing Foot Kd: \n" << K_d_swing_foot << std::endl;
    std::cout << "\n";
  }

  // Get body frames and points
  Vector3d mid_contact_point = (left_toe.first + left_heel.first) / 2;
  auto left_toe_mid = std::pair<const Vector3d, const Frame<double>&>(
      mid_contact_point, plant_ctrl.GetFrameByName("toe_left"));
  auto right_toe_mid = std::pair<const Vector3d, const Frame<double>&>(
      mid_contact_point, plant_ctrl.GetFrameByName("toe_right"));
  auto left_toe_origin = std::pair<const Vector3d, const Frame<double>&>(
      Vector3d::Zero(), plant_ctrl.GetFrameByName("toe_left"));
  auto right_toe_origin = std::pair<const Vector3d, const Frame<double>&>(
      Vector3d::Zero(), plant_ctrl.GetFrameByName("toe_right"));

  // Create state receiver.
  auto state_receiver =
      builder.AddSystem<systems::RobotOutputReceiver>(plant_ctrl);
  if (FLAGS_use_dispatcher) {
    builder.Connect(robot_output_sender->get_output_port(0),
                    state_receiver->get_input_port(0));
  } else {
    builder.Connect(state_sender->get_output_port(0),
                    state_receiver->get_input_port(0));
  }

  // Create command sender.
  auto command_pub =
      builder.AddSystem(LcmPublisherSystem::Make<dairlib::lcmt_robot_input>(
          FLAGS_channel_u + suffix, &lcm_local, 1.0 / FLAGS_publish_rate));
  auto command_sender =
      builder.AddSystem<systems::RobotCommandSender>(plant_ctrl);
  builder.Connect(command_sender->get_output_port(0),
                  input_receiver->get_input_port());
  builder.Connect(command_sender->get_output_port(0),
                  command_pub->get_input_port());

  // Add emulator for floating base drift
  Eigen::VectorXd drift_mean =
      Eigen::VectorXd::Zero(plant_ctrl.num_positions());
  Eigen::MatrixXd drift_cov = Eigen::MatrixXd::Zero(plant_ctrl.num_positions(),
                                                    plant_ctrl.num_positions());
  drift_cov(4, 4) = FLAGS_drift_rate;  // x
  drift_cov(5, 5) = FLAGS_drift_rate;  // y
  drift_cov(6, 6) = FLAGS_drift_rate;  // z
  // Note that we didn't add drift to yaw angle here because it requires
  // changing SimulatorDrift.

  auto simulator_drift =
      builder.AddSystem<SimulatorDrift>(plant_ctrl, drift_mean, drift_cov);
  builder.Connect(state_receiver->get_output_port(0),
                  simulator_drift->get_input_port_state());

  // Create human high-level control
  Eigen::Vector2d global_target_position(0, 0);
  Eigen::Vector2d params_of_no_turning(5, 1);
  // Logistic function 1/(1+5*exp(x-1))
  // The function ouputs 0.0007 when x = 0
  //                     0.5    when x = 1
  //                     0.9993 when x = 2
  cassie::osc::HighLevelCommand* high_level_command;
  if (FLAGS_use_radio) {
    //    auto cassie_out_receiver =
    //        builder.AddSystem(LcmSubscriberSystem::Make<dairlib::lcmt_cassie_out>(
    //            FLAGS_cassie_out_channel + suffix, &lcm_local));
    //    double vel_scale_rot = 0.5;
    //    double vel_scale_trans = 1.5;
    //    high_level_command = builder.AddSystem<cassie::osc::HighLevelCommand>(
    //        plant_ctrl, context_w_spr.get(), vel_scale_rot, vel_scale_trans,
    //        FLAGS_footstep_option);
    //    builder.Connect(cassie_out_receiver->get_output_port(),
    //                    high_level_command->get_cassie_output_port());
  } else {
    high_level_command = builder.AddSystem<cassie::osc::HighLevelCommand>(
        plant_ctrl, context_w_spr.get(), global_target_position,
        params_of_no_turning, FLAGS_footstep_option);
  }
  builder.Connect(state_receiver->get_output_port(0),
                  high_level_command->get_state_input_port());

  // Create heading traj generator
  auto head_traj_gen = builder.AddSystem<cassie::osc::HeadingTrajGenerator>(
      plant_ctrl, context_w_spr.get());
  builder.Connect(simulator_drift->get_output_port(0),
                  head_traj_gen->get_state_input_port());
  builder.Connect(high_level_command->get_yaw_output_port(),
                  head_traj_gen->get_yaw_input_port());

  // Create finite state machine
  int left_stance_state = 0;
  int right_stance_state = 1;
  int double_support_state = 2;
  double left_support_duration = 0.35;
  double right_support_duration = 0.35;
  double double_support_duration = gains.double_support_duration;
  vector<int> fsm_states;
  vector<double> state_durations;
  if (FLAGS_is_two_phase) {
    fsm_states = {left_stance_state, right_stance_state};
    state_durations = {left_support_duration, right_support_duration};
  } else {
    fsm_states = {left_stance_state, double_support_state, right_stance_state,
                  double_support_state};
    state_durations = {left_support_duration, double_support_duration,
                       right_support_duration, double_support_duration};
  }
  auto fsm = builder.AddSystem<systems::TimeBasedFiniteStateMachine>(
      plant_ctrl, fsm_states, state_durations);
  builder.Connect(simulator_drift->get_output_port(0),
                  fsm->get_input_port_state());

  // Create leafsystem that record the switching time of the FSM
  std::vector<int> single_support_states = {left_stance_state,
                                            right_stance_state};
  auto liftoff_event_time =
      builder.AddSystem<systems::FiniteStateMachineEventTime>(
          single_support_states);
  liftoff_event_time->set_name("liftoff_time");
  builder.Connect(fsm->get_output_port(0),
                  liftoff_event_time->get_input_port_fsm());
  auto touchdown_event_time =
      builder.AddSystem<systems::FiniteStateMachineEventTime>(
          std::vector<int>(1, double_support_state));
  touchdown_event_time->set_name("touchdown_time");
  builder.Connect(fsm->get_output_port(0),
                  touchdown_event_time->get_input_port_fsm());

  // Create CoM trajectory generator
  // Note that we are tracking COM acceleration instead of position and velocity
  // because we construct the LIPM traj which starts from the current state
  double desired_com_height = gains.lipm_height;
  vector<int> unordered_fsm_states;
  vector<double> unordered_state_durations;
  vector<vector<std::pair<const Vector3d, const Frame<double>&>>>
      contact_points_in_each_state;
  if (FLAGS_is_two_phase) {
    unordered_fsm_states = {left_stance_state, right_stance_state};
    unordered_state_durations = {left_support_duration, right_support_duration};
    contact_points_in_each_state.push_back({left_toe_mid});
    contact_points_in_each_state.push_back({right_toe_mid});
  } else {
    unordered_fsm_states = {left_stance_state, right_stance_state,
                            double_support_state};
    unordered_state_durations = {left_support_duration, right_support_duration,
                                 double_support_duration};
    contact_points_in_each_state.push_back({left_toe_mid});
    contact_points_in_each_state.push_back({right_toe_mid});
    contact_points_in_each_state.push_back({left_toe_mid, right_toe_mid});
  }
  auto lipm_traj_generator = builder.AddSystem<systems::LIPMTrajGenerator>(
      plant_ctrl, context_w_spr.get(), desired_com_height, unordered_fsm_states,
      unordered_state_durations, contact_points_in_each_state);
  builder.Connect(fsm->get_output_port(0),
                  lipm_traj_generator->get_input_port_fsm());
  builder.Connect(touchdown_event_time->get_output_port_event_time(),
                  lipm_traj_generator->get_input_port_fsm_switch_time());
  builder.Connect(simulator_drift->get_output_port(0),
                  lipm_traj_generator->get_input_port_state());

  // Create velocity control by foot placement
  bool use_predicted_com_vel = true;
  auto walking_speed_control =
      builder.AddSystem<cassie::osc::WalkingSpeedControl>(
          plant_ctrl, context_w_spr.get(), FLAGS_footstep_option,
          use_predicted_com_vel ? left_support_duration : 0);
  builder.Connect(high_level_command->get_xy_output_port(),
                  walking_speed_control->get_input_port_des_hor_vel());
  builder.Connect(simulator_drift->get_output_port(0),
                  walking_speed_control->get_input_port_state());
  if (use_predicted_com_vel) {
    builder.Connect(lipm_traj_generator->get_output_port_lipm_from_current(),
                    walking_speed_control->get_input_port_com());
    builder.Connect(
        liftoff_event_time->get_output_port_event_time_of_interest(),
        walking_speed_control->get_input_port_fsm_switch_time());
  }

  // Create swing leg trajectory generator (capture point)
  // Since the ground is soft in the simulation, we raise the desired final
  // foot height by 1 cm. The controller is sensitive to this number, should
  // tune this every time we change the simulation parameter or when we move
  // to the hardware testing.
  // Additionally, implementing a double support phase might mitigate the
  // instability around state transition.
  double max_CoM_to_footstep_dist = 0.4;

  vector<int> left_right_support_fsm_states = {left_stance_state,
                                               right_stance_state};
  vector<double> left_right_support_state_durations = {left_support_duration,
                                                       right_support_duration};
  vector<std::pair<const Vector3d, const Frame<double>&>> left_right_foot = {
      left_toe_origin, right_toe_origin};
  auto swing_ft_traj_generator =
      builder.AddSystem<systems::SwingFootTrajGenerator>(
          plant_ctrl, context_w_spr.get(), left_right_support_fsm_states,
          left_right_support_state_durations, left_right_foot, "pelvis",
          gains.mid_foot_height, gains.final_foot_height,
          gains.final_foot_velocity_z, max_CoM_to_footstep_dist,
          gains.footstep_offset, gains.center_line_offset, true, true, true,
          FLAGS_footstep_option);
  builder.Connect(fsm->get_output_port(0),
                  swing_ft_traj_generator->get_input_port_fsm());
  builder.Connect(liftoff_event_time->get_output_port_event_time_of_interest(),
                  swing_ft_traj_generator->get_input_port_fsm_switch_time());
  builder.Connect(simulator_drift->get_output_port(0),
                  swing_ft_traj_generator->get_input_port_state());
  builder.Connect(lipm_traj_generator->get_output_port_lipm_from_current(),
                  swing_ft_traj_generator->get_input_port_com());
  builder.Connect(walking_speed_control->get_output_port(0),
                  swing_ft_traj_generator->get_input_port_sc());

  // Create Operational space control
  auto osc = builder.AddSystem<systems::controllers::OperationalSpaceControl>(
      plant_ctrl, plant_ctrl, context_w_spr.get(), context_w_spr.get(), true,
      FLAGS_print_osc /*print_tracking_info*/);

  // Cost
  int n_v = plant_ctrl.num_velocities();
  MatrixXd Q_accel = gains.w_accel * MatrixXd::Identity(n_v, n_v);
  osc->SetAccelerationCostForAllJoints(Q_accel);

  // Constraints in OSC
  multibody::KinematicEvaluatorSet<double> evaluators(plant_ctrl);
  // 1. fourbar constraint
  evaluators.add_evaluator(&left_loop);
  evaluators.add_evaluator(&right_loop);
  // 2. fixed spring constriant
  // Note that we set the position value to 0, but this is not used in OSC,
  // because OSC constraint only use JdotV and J.
  auto pos_idx_map = multibody::makeNameToPositionsMap(plant_ctrl);
  auto vel_idx_map = multibody::makeNameToVelocitiesMap(plant_ctrl);
  auto left_fixed_knee_spring =
      FixedJointEvaluator(plant_ctrl, pos_idx_map.at("knee_joint_left"),
                          vel_idx_map.at("knee_joint_leftdot"), 0);
  auto right_fixed_knee_spring =
      FixedJointEvaluator(plant_ctrl, pos_idx_map.at("knee_joint_right"),
                          vel_idx_map.at("knee_joint_rightdot"), 0);
  auto left_fixed_ankle_spring =
      FixedJointEvaluator(plant_ctrl, pos_idx_map.at("ankle_spring_joint_left"),
                          vel_idx_map.at("ankle_spring_joint_leftdot"), 0);
  auto right_fixed_ankle_spring = FixedJointEvaluator(
      plant_ctrl, pos_idx_map.at("ankle_spring_joint_right"),
      vel_idx_map.at("ankle_spring_joint_rightdot"), 0);
  evaluators.add_evaluator(&left_fixed_knee_spring);
  evaluators.add_evaluator(&right_fixed_knee_spring);
  evaluators.add_evaluator(&left_fixed_ankle_spring);
  evaluators.add_evaluator(&right_fixed_ankle_spring);
  osc->AddKinematicConstraint(&evaluators);

  // Soft constraint
  // w_contact_relax shouldn't be too big, cause we want tracking error to be
  // important
  osc->SetWeightOfSoftContactConstraint(gains.w_soft_constraint);
  // Friction coefficient
  double mu = 0.4;
  osc->SetContactFriction(mu);
  // Add contact points (The position doesn't matter. It's not used in OSC)
  osc->AddStateAndContactPoint(left_stance_state, &left_toe_evaluator);
  osc->AddStateAndContactPoint(left_stance_state, &left_heel_evaluator);
  osc->AddStateAndContactPoint(right_stance_state, &right_toe_evaluator);
  osc->AddStateAndContactPoint(right_stance_state, &right_heel_evaluator);
  if (!FLAGS_is_two_phase) {
    osc->AddStateAndContactPoint(double_support_state, &left_toe_evaluator);
    osc->AddStateAndContactPoint(double_support_state, &left_heel_evaluator);
    osc->AddStateAndContactPoint(double_support_state, &right_toe_evaluator);
    osc->AddStateAndContactPoint(double_support_state, &right_heel_evaluator);
  }

  // Swing foot tracking
  TransTaskSpaceTrackingData swing_foot_traj("swing_ft_traj", K_p_swing_foot,
                                             K_d_swing_foot, W_swing_foot,
                                             plant_ctrl, plant_ctrl);
  swing_foot_traj.AddStateAndPointToTrack(left_stance_state, "toe_right");
  swing_foot_traj.AddStateAndPointToTrack(right_stance_state, "toe_left");
  osc->AddTrackingData(&swing_foot_traj);
  // Center of mass tracking
  //  ComTrackingData center_of_mass_traj("lipm_traj", K_p_com, K_d_com, W_com,
  //                                      plant_ctrl, plant_ctrl);
  TransTaskSpaceTrackingData center_of_mass_traj("lipm_traj", K_p_com, K_d_com,
                                                 W_com, plant_ctrl, plant_ctrl);
  center_of_mass_traj.AddPointToTrack("pelvis");
  osc->AddTrackingData(&center_of_mass_traj);
  // Pelvis rotation tracking (pitch and roll)
  RotTaskSpaceTrackingData pelvis_balance_traj(
      "pelvis_balance_traj", K_p_pelvis_balance, K_d_pelvis_balance,
      W_pelvis_balance, plant_ctrl, plant_ctrl);
  pelvis_balance_traj.AddFrameToTrack("pelvis");
  VectorXd pelvis_desired_quat(4);
  pelvis_desired_quat << 1, 0, 0, 0;
  osc->AddConstTrackingData(&pelvis_balance_traj, pelvis_desired_quat);
  // Pelvis rotation tracking (yaw)
  RotTaskSpaceTrackingData pelvis_heading_traj(
      "pelvis_heading_traj", K_p_pelvis_heading, K_d_pelvis_heading,
      W_pelvis_heading, plant_ctrl, plant_ctrl);
  pelvis_heading_traj.AddFrameToTrack("pelvis");
  osc->AddTrackingData(&pelvis_heading_traj, 0.1);  // 0.05
  // Swing toe joint tracking (Currently use fix position)
  // The desired position, -1.5, was derived heuristically. It is roughly the
  // toe angle when Cassie stands on the ground.
  MatrixXd W_swing_toe = gains.w_swing_toe * MatrixXd::Identity(1, 1);
  MatrixXd K_p_swing_toe = gains.swing_toe_kp * MatrixXd::Identity(1, 1);
  MatrixXd K_d_swing_toe = gains.swing_toe_kd * MatrixXd::Identity(1, 1);
  JointSpaceTrackingData swing_toe_traj("swing_toe_traj", K_p_swing_toe,
                                        K_d_swing_toe, W_swing_toe, plant_ctrl,
                                        plant_ctrl);
  swing_toe_traj.AddStateAndJointToTrack(left_stance_state, "toe_right",
                                         "toe_rightdot");
  swing_toe_traj.AddStateAndJointToTrack(right_stance_state, "toe_left",
                                         "toe_leftdot");
  osc->AddConstTrackingData(&swing_toe_traj, -1.5 * VectorXd::Ones(1), 0, 0.3);
  // Swing hip yaw joint tracking
  MatrixXd W_hip_yaw = gains.w_hip_yaw * MatrixXd::Identity(1, 1);
  MatrixXd K_p_hip_yaw = gains.hip_yaw_kp * MatrixXd::Identity(1, 1);
  MatrixXd K_d_hip_yaw = gains.hip_yaw_kd * MatrixXd::Identity(1, 1);
  JointSpaceTrackingData swing_hip_yaw_traj("swing_hip_yaw_traj", K_p_hip_yaw,
                                            K_d_hip_yaw, W_hip_yaw, plant_ctrl,
                                            plant_ctrl);
  swing_hip_yaw_traj.AddStateAndJointToTrack(left_stance_state, "hip_yaw_right",
                                             "hip_yaw_rightdot");
  swing_hip_yaw_traj.AddStateAndJointToTrack(right_stance_state, "hip_yaw_left",
                                             "hip_yaw_leftdot");
  osc->AddConstTrackingData(&swing_hip_yaw_traj, VectorXd::Zero(1));
  // Build OSC problem
  osc->Build();
  // Connect ports
  builder.Connect(simulator_drift->get_output_port(0),
                  osc->get_robot_output_input_port());
  builder.Connect(fsm->get_output_port(0), osc->get_fsm_input_port());
  builder.Connect(lipm_traj_generator->get_output_port_lipm_from_touchdown(),
                  osc->get_tracking_data_input_port("lipm_traj"));
  builder.Connect(swing_ft_traj_generator->get_output_port(0),
                  osc->get_tracking_data_input_port("swing_ft_traj"));
  builder.Connect(head_traj_gen->get_output_port(0),
                  osc->get_tracking_data_input_port("pelvis_heading_traj"));
  builder.Connect(osc->get_output_port(0), command_sender->get_input_port(0));
  if (FLAGS_publish_osc_data) {
    // Create osc debug sender.
    auto osc_debug_pub =
        builder.AddSystem(LcmPublisherSystem::Make<dairlib::lcmt_osc_output>(
            "OSC_DEBUG_WALKING" + suffix, &lcm_local,
            1.0 / FLAGS_publish_rate));
    builder.Connect(osc->get_osc_debug_port(), osc_debug_pub->get_input_port());
  }

  //////////////////// Build the whole diagram ////////////////////////
  auto diagram = builder.Build();

  // Create a context for this system:
  std::unique_ptr<Context<double>> diagram_context =
      diagram->CreateDefaultContext();
  diagram_context->EnableCaching();
  diagram->SetDefaultContext(diagram_context.get());
  Context<double>& plant_context =
      diagram->GetMutableSubsystemContext(plant_sim, diagram_context.get());

  // Set initial conditions of the simulation
  VectorXd q_init, u_init, lambda_init;
  double mu_fp = 0;
  double min_normal_fp = 70;
  double toe_spread = .2;
  // Create a plant for CassieFixedPointSolver.
  // Note that we cannot use the plant from the above diagram, because after the
  // diagram is built, plant_sim.get_actuation_input_port().HasValue(*context)
  // throws a segfault error
  drake::multibody::MultibodyPlant<double> plant_for_solver(0.0);
  addCassieMultibody(&plant_for_solver, nullptr, true /*floating base*/, urdf,
                     true, true);
  plant_for_solver.Finalize();
  CassieFixedPointSolver(plant_for_solver, FLAGS_init_height, mu_fp,
                         min_normal_fp, true, toe_spread, &q_init, &u_init,
                         &lambda_init);
  VectorXd v_init = VectorXd::Zero(plant_sim.num_velocities());
  v_init(3) = FLAGS_pelvis_disturbnace_xdot;
  v_init(4) = FLAGS_pelvis_disturbnace_ydot;
  v_init(5) = FLAGS_pelvis_disturbnace_zdot;
  plant_sim.SetPositions(&plant_context, q_init);
  plant_sim.SetVelocities(&plant_context, v_init);

  // Set EKF time and initial states
  setInitialEkfState(0, q_init, *diagram, *state_estimator,
                     diagram_context.get());

  Simulator<double> simulator(*diagram, std::move(diagram_context));

  if (!FLAGS_time_stepping) {
    // simulator.get_mutable_integrator()->set_maximum_step_size(0.01);
    // simulator.get_mutable_integrator()->set_target_accuracy(1e-1);
    // simulator.get_mutable_integrator()->set_fixed_step_mode(true);
    simulator.reset_integrator<drake::systems::RungeKutta2Integrator<double>>(
        FLAGS_dt);
  }

  simulator.set_publish_every_time_step(false);
  simulator.set_publish_at_initialization(false);
  simulator.set_target_realtime_rate(FLAGS_target_realtime_rate);
  //  cout << "initialize\n";
  simulator.Initialize();
  //  cout << "advanceto\n";
  simulator.AdvanceTo(FLAGS_end_time);
  //  cout << "finished simulating\n";

  return 0;
}

}  // namespace dairlib

int main(int argc, char* argv[]) { return dairlib::DoMain(argc, argv); }
