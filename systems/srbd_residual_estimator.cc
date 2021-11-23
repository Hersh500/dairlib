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
                                                 plant_(plant),
                                                 rate_(rate),
                                                 buffer_len_(buffer_len),
                                                 use_fsm_(use_fsm) {

      // Initialize data matrices
      X_ = MatrixXd::Zero(buffer_len_, num_X_cols);
      y_ = MatrixXd::Zero(buffer_len_, nx_);

      // Declare all ports
      state_in_port_ = this->DeclareVectorInputPort(
                      "x, u, t",
                      OutputVector<double>(plant_.nq(), plant_.nv(), plant_.nu()))
              .get_index();

      A_hat_port_ = this->DeclareAbstractOutputPort("A_hat",
                                                    &SRBDResidualEstimator::GetAHat).get_index();
      B_hat_port_ = this->DeclareAbstractOutputPort("B_hat",
                                                    &SRBDResidualEstimator::GetBHat).get_index();
      b_hat_port_ = this->DeclareAbstractOutputPort("b_hat",
                                                    &SRBDResidualEstimator::GetbHat).get_index();


      if ( use_fsm_ ) {
        fsm_port_ = this->DeclareVectorInputPort(
                        "fsm",
                        BasicVector<double>(1))
                .get_index();

        current_fsm_state_idx_ =
                this->DeclareDiscreteState(VectorXd::Zero(1));
        prev_event_time_idx_ = this->DeclareDiscreteState(-0.1 * VectorXd::Ones(1));
      }


      DeclarePerStepDiscreteUpdateEvent(&SRBDResidualEstimator::DiscreteVariableUpdate);
      DeclarePeriodicDiscreteUpdateEvent(rate_, 0, &SRBDResidualEstimator::PeriodicUpdate);

    }

    void SRBDResidualEstimator::GetAHat(const drake::systems::Context<double> &context, Eigen::MatrixXd *A_msg) const {
      *A_msg = cur_A_hat_;
    }

    void SRBDResidualEstimator::GetBHat(const drake::systems::Context<double> &context, Eigen::MatrixXd *B_msg) const {
      *B_msg = cur_B_hat_;
    }

    void SRBDResidualEstimator::GetbHat(const drake::systems::Context<double> &context, Eigen::MatrixXd *b_msg) const {
      *b_msg = cur_b_hat_;
    }

    void SRBDResidualEstimator::AddMode(const LinearSrbdDynamics&  dynamics,
                           BipedStance stance, const MatrixXd& reset, int N) {
      DRAKE_DEMAND(stance == nmodes_);
      SrbdMode mode = {dynamics, reset, stance, N};
      modes_.push_back(mode);
      nmodes_++;
    }

    drake::systems::EventStatus SRBDResidualEstimator::PeriodicUpdate(
            const drake::systems::Context<double> &context,
            drake::systems::DiscreteValues<double>* discrete_state) const {
      // Solve least squares only if it's ready.
      if (ticks_ > buffer_len_) {
        SolveLstSq();
      }
      return drake::systems::EventStatus::Succeeded();
    }

    // For now, calling this as a discrete variable update though it doesn't have to be.
    drake::systems::EventStatus SRBDResidualEstimator::DiscreteVariableUpdate(const drake::systems::Context<double> &context,
                                                  drake::systems::DiscreteValues<double> *discrete_state) const {
      const OutputVector<double>* robot_output =
              (OutputVector<double>*)this->EvalVectorInput(context, state_in_port_);

      double timestamp = robot_output->get_timestamp();

      // FSM stuff (copied from the srbd_mpc)
      const BasicVector<double>* fsm_output =
              (BasicVector<double>*)this->EvalVectorInput(context, fsm_port_);
      VectorXd fsm_state = fsm_output->get_value();
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

      // Full robot state
      VectorXd x = robot_output->GetState();

      // Get the srbd state and foot state
      VectorXd srbd_state = plant_.CalcSRBStateFromPlantState(x);

      // switch based on the contact state.
      std::vector<Eigen::Vector3d> foot_locs = plant_.CalcFootPositions(x);

      Eigen::Vector3d foot_loc = foot_locs[fsm_state(0)];
      BipedStance cur_stance_mode = modes_.at(fsm_state(0)).stance;

      // TODO(hersh500): get the u by multiplying the jacobian with efforts, then selecting the appropriate column.
      Eigen::VectorXd u = Eigen::VectorXd::Zero(nu_);

      UpdateLstSqEquation(srbd_state, u, foot_loc, cur_stance_mode);
      if (ticks_ < buffer_len_) {
        ticks_++;
      }
      return drake::systems::EventStatus::Succeeded();
    }

    void SRBDResidualEstimator::UpdateLstSqEquation(Eigen::VectorXd state,
                                                    Eigen::VectorXd input,
                                                    Eigen::Vector3d stance_foot_loc,
                                                    BipedStance stance_mode) const {
      VectorXd vec_joined(nx_ + 3);
      vec_joined << state, stance_foot_loc;

      // Rotate X and y up by a row.
      X_.block(0, 0, buffer_len_ - 1, num_X_cols) = X_.block(1, 0, buffer_len_ - 1, num_X_cols);
      y_.block(0, 0, buffer_len_ - 1, nx_) = y_.block(1, 0, buffer_len_ - 1, nx_);

      // set the last row of X to the current state and input, and ones
      X_.row(buffer_len_ - 1).head(nx_ + 3) = vec_joined;
      X_.row(buffer_len_ - 1).segment(nx_ + 3, nu_) = input;
      X_.row(buffer_len_ - 1).tail(nx_) = Eigen::VectorXd::Ones(nx_);

      // Set the last row of Y to be the -A * cur_state - B * cur_input - b (based on the stance mode)
      // the last row isn't ready to use yet since the next state needs to be added to it.
      // Implicit cast from stance_mode (BipedStance) to int isn't great.
      y_.row(buffer_len_ - 1) = -modes_.at(stance_mode).dynamics.A * vec_joined - -modes_.at(stance_mode).dynamics.B * input - -modes_.at(stance_mode).dynamics.b;


      // Add the current state to the 2nd to last row of Y, completing the temporary part.
      y_.row(buffer_len_ - 2) = y_.row(buffer_len_-2) + state;
    }

    void SRBDResidualEstimator::SolveLstSq() const {
      // Solve the least squares equation, excluding the last row of X and y because it is always incomplete.
      Eigen::MatrixXd X_c = X_.block(0, 0, buffer_len_-1, num_X_cols);
      Eigen::MatrixXd y_c = y_.block(0, 0, buffer_len_-1, nx_);

      Eigen::MatrixXd soln = X_c.colPivHouseholderQr().solve(y_c).transpose();

      // Select the appropriate parts of the solution matrix for the residuals
      cur_A_hat_ = soln.block(0, 0, nx_, nx_+3);
      cur_B_hat_ = soln.block(0, nx_ + 3, nx_, nu_);
      cur_b_hat_ = soln.block(0, nx_ + nu_ + 3, nx_, 1);
    }
}