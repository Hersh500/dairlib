//
// Created by hersh on 11/15/21.
//

#ifndef DAIRLIB_SRBD_RESIDUAL_ESTIMATOR_H
#define DAIRLIB_SRBD_RESIDUAL_ESTIMATOR_H
#include "drake/systems/framework/leaf_system.h"
#include "drake/solvers/mathematical_program.h"

#include "multibody/single_rigid_body_plant.h"

namespace dairlib {
class SRBDResidualEstimator : public drake::systems::LeafSystem<double> {
public:
    SRBDResidualEstimator(const multibody::SingleRigidBodyPlant& plant, double rate, unsigned int buffer_len);

    // Want to connect this to a callback that adds the state to a deque
    const drake::systems::InputPort<double>& get_state_input_port() const {
        return this->get_input_port(state_in_port_);
    };

    const drake::systems::OutputPort<double>& get_A_hat_output_port() const {
        return this->get_output_port(A_hat_port_);
    };

    const drake::systems::OutputPort<double>& get_B_hat_output_port() const {
      return this->get_output_port(B_hat_port_);
    };

    const drake::systems::OutputPort<double>& get_b_hat_output_port() const {
      return this->get_output_port(b_hat_port_);
    };

    unsigned int buffer_len_;
    double rate_;

private:
    // states from estimator get added to this, used to build least squares problem
    Eigen::MatrixXd X_;

    // Transition states for least squares estimator.
    Eigen::MatrixXd y_;

    // TODO(hersh500): use LinearSRBDDynamics struct for this
    // Output matrices
    Eigen::MatrixXd cur_A_hat_, cur_B_hat_, cur_b_hat_;

    // Nominal dynamics
    Eigen::MatrixXd A, B, b;

    int state_in_port_, A_hat_port_, B_hat_port_, b_hat_port_, fsm_port_;
    int ticks = 0;
    std::pair<int, int> A_dim = std::pair<int, int>(12, 15);
    std::pair<int, int> B_dim = std::pair<int, int>(12, 4);
    int b_dim = 12;

    bool use_fsm_;

    // keep this plant to use its utility functions like getting the dynamics, etc.
    const multibody::SingleRigidBodyPlant& plant_;

    // Solves the Least Squares Problem, connects matrices to outputs
    drake::systems::EventStatus PeriodicUpdate(
            const drake::systems::Context<double> &context,
            drake::systems::DiscreteValues<double>* discrete_state) const;

    // Only gets called when a new LCM message is present, since it's wrapped in an LCM driven loop.
    drake::systems::EventStatus DiscreteVariableUpdate(
            const drake::systems::Context<double>& context,
            drake::systems::DiscreteValues<double>* discrete_state) const;

    // Assume that the state is the SRBD state
    void UpdateLstSqEquation(Eigen::VectorXd state, Eigen::VectorXd stance_foot_loc,
                             BipedStance stance_mode);
};

}
#endif //DAIRLIB_SRBD_RESIDUAL_ESTIMATOR_H
