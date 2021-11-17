//
// Created by hersh on 11/15/21.
//
#include "srbd_residual_estimator.h"
#include "systems/framework/output_vector.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using drake::systems::BasicVector;
using dairlib::systems::OutputVector;

namespace dairlib {
    SRBDResidualEstimator::SRBDResidualEstimator(const multibody::SingleRigidBodyPlant &plant, double rate,
                                                 unsigned int buffer_len, bool use_fsm) :
                                                 buffer_len_(buffer_len),
                                                 rate_(rate),
                                                 plant_(plant),
                                                 use_fsm_(use_fsm) {

      // Initialize data matrices
      X_ = MatrixXd::Zero(buffer_len_, plant_.nq()+plant_.nu());
      y_ = MatrixXd::Zero(buffer_len_, plant_.nq());

      // Declare all ports
      state_in_port_ = this->DeclareVectorInputPort(
                      "x, u, t",
                      OutputVector<double>(plant_.nq(), plant_.nv(), plant_.nu()))
              .get_index();

      if ( use_fsm_ ) {
        fsm_port_ = this->DeclareVectorInputPort(
                        "fsm",
                        BasicVector<double>(1))
                .get_index();

        current_fsm_state_idx_ =
                this->DeclareDiscreteState(VectorXd::Zero(1));
        prev_event_time_idx_ = this->DeclareDiscreteState(-0.1 * VectorXd::Ones(1));
      }


//      DeclarePerStepDiscreteUpdateEvent(&SRBDResidualEstimator::DiscreteVariableUpdate);
//      DeclarePeriodicDiscreteUpdateEvent(rate_, 0, &SRBDResidualEstimator::PeriodicUpdate);

    }
    // TODO: actually have to declare this event to happen somehow?
    drake::systems::EventStatus SRBDResidualEstimator::DiscreteVariableUpdate(
            const drake::systems::Context<double> &context,
            drake::systems::DiscreteValues<double> *discrete_state) const {

      const OutputVector<double>* robot_output =
      (OutputVector<double>*)this->EvalVectorInput(context, state_in_port_);
      double timestamp = robot_output->get_timestamp();

      if (use_fsm_) {
        auto current_fsm_state =
                discrete_state->get_mutable_vector(current_fsm_state_idx_)
                        .get_mutable_value();

        if (fsm_state(0) != current_fsm_state(0)) {
          current_fsm_state(0) = fsm_state(0);
          discrete_state->get_mutable_vector(prev_event_time_idx_).get_mutable_value()
                  << timestamp;
        }
      }
      return EventStatus::Succeeded();
    }

    void SRBDResidualEstimator::UpdateLstSqEquation(Eigen::VectorXd state,
                                                    Eigen::VectorXd stance_foot_loc,
                                                    BipedStance stance_mode) {

    }




}