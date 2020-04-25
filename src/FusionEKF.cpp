#include "FusionEKF.h"
#include <iostream>
#include "Eigen/Dense"
#include "tools.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::cout;
using std::endl;
using std::vector;

/**
 * Constructor.
 */
FusionEKF::FusionEKF() {
  is_initialized_ = false;

  previous_timestamp_ = 0;

  // initializing matrices
  R_laser_ = MatrixXd(2, 2);
  R_radar_ = MatrixXd(3, 3);
  H_laser_ = MatrixXd(2, 4);

  //measurement covariance matrix - laser
  R_laser_ << 0.0225, 0,
              0, 0.0225;

  //measurement covariance matrix - radar
  R_radar_ << 0.09, 0, 0,
              0, 0.0009, 0,
              0, 0, 0.09;

  // hidden -> measurement matrix H - laser
  H_laser_ << 1, 0, 0, 0,
              0, 1, 0, 0;

  /**
   * TODO: Finish initializing the FusionEKF.
   * TODO: Set the process and measurement noises
   */
  ekf_.x_ = VectorXd(4);
  ekf_.P_ = MatrixXd(4, 4);
  ekf_.F_ = MatrixXd(4, 4);
  ekf_.F_ << 1, 0, 1, 0,
            0, 1, 0, 1,
            0, 0, 1, 0,
            0, 0, 0, 1;
  noise_ax_ = 9;
  noise_ay_ = 9;


}

/**
 * Destructor.
 */
FusionEKF::~FusionEKF() {}

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) {
  /**
   * Initialization
   */
  if (!is_initialized_) {
    // first measurement
    cout << "EKF: " << endl;
    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
      float rho = measurement_pack.raw_measurements_(0);
      float phi = measurement_pack.raw_measurements_(1);
      float rho_dot = measurement_pack.raw_measurements_(2);
      
      float px = rho * cos(phi);
      float py = rho * sin(phi);

      float vx = rho_dot * cos(phi);
      float vy = rho_dot * sin(phi);
      
      ekf_.x_ << px, py, vx, vy;

    } else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
      float px = measurement_pack.raw_measurements_(0);
      float py = measurement_pack.raw_measurements_(1);
      ekf_.x_ << px, py, 0, 0;
    }

    // Initial P covar matrix
    ekf_.P_ << 1, 0, 0, 0,
              0, 1, 0, 0,
              0, 0, 1000, 0,
              0, 0, 0, 1000;

    // done initializing, no need to predict or update
    is_initialized_ = true;
    previous_timestamp_ = measurement_pack.timestamp_;
    return;
  }

  /**
   * Prediction
   */
  float dt = (measurement_pack.timestamp_ - previous_timestamp_) / 1000000.0;
  previous_timestamp_ = measurement_pack.timestamp_;

  // Define matrix F
  ekf_.F_(0, 2) = dt;
  ekf_.F_(1, 3) = dt;

  // Covar matrix stochastic Q
  float dt2 = dt * dt;
  float dt3 = dt2 * dt / 2;
  float dt4 = dt3 * dt / 4;
  ekf_.Q_ = MatrixXd(4, 4);
  ekf_.Q_ << dt4 * noise_ax_,   0,                  dt3 * noise_ax_,    0,
             0,                 dt4 * noise_ay_,    0,                  dt3 * noise_ay_,
             dt3 * noise_ax_,   0,                  dt2 * noise_ax_,    0,
             0,                 dt3 * noise_ay_,    0,                  dt2 * noise_ay_;
  ekf_.Predict();

  /**
   * Update
   */
  if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
    // Radar updates

    // z_pred
    VectorXd z_pred = VectorXd(3);
    float rho = sqrt(ekf_.x_(0) * ekf_.x_(0) + ekf_.x_(1) * ekf_.x_(1));
    float phi = atan2(ekf_.x_(1), ekf_.x_(0));
    float rho_dot = ((ekf_.x_(0) * ekf_.x_(2)) + (ekf_.x_(1) * ekf_.x_(3))) / rho;
    z_pred << rho, phi, rho_dot;

    // Define Kalman matrices for Radar
    ekf_.H_ = Tools::CalculateJacobian(ekf_.x_);
    ekf_.R_ = R_radar_;

    // update
    ekf_.UpdateEKF(measurement_pack.raw_measurements_, z_pred);

  } else {
    // Laser updates
    // z_pred will compute inside of Update method.
    // Define Kalman matrices for Laser
    ekf_.H_ = H_laser_;
    ekf_.R_ = R_laser_;

    // update
    ekf_.Update(measurement_pack.raw_measurements_);
  }

  // print the output
  cout << "x_ = " << ekf_.x_ << endl;
  cout << "P_ = " << ekf_.P_ << endl;
}
