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
    SRBDResidualEstimator(const multibody::SingleRigidBodyPlant& plant, double dt, int buffer_len);

    // Want to connect this to a callback that adds the state to a deque
    const drake::systems::InputPort<double>& get_state_input_port() const {
        return this->get_input_port(state_in_port_);
    };

    const drake::systems::OutputPort<double>& get_A_hat_output_port() const {
        return this->get_output_port(A_hat_port_);
    };


private:
    // states from estimator get added to this, used to build least squares problem
    std::deque<Eigen::VectorXd> state_buffer;
    int state_in_port_;
    int A_hat_port_;
    int B_hat_port_;
    int b_hat_port_;

    // keep this plant to use its utility functions
    const multibody::SingleRigidBodyPlant& plant_;

    // Solves the Least Squares Problem, connects matrices to outputs
    drake::systems::EventStatus PeriodicUpdate(
            const drake::systems::Context<double> &context,
            drake::systems::DiscreteValues<double>* discrete_state);

};

}
#endif //DAIRLIB_SRBD_RESIDUAL_ESTIMATOR_H
