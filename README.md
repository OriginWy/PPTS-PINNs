# PPTS-PINNs

The project is developed using Python 3.10 and PyTorch 2.5, and includes 12 subprojects, organized as follows:

1. Project Structure

1. mPINNs:
   - mPINNs-R1
   - mPINNs-R1-2
   - mPINNs-R1-3
   - mPINNs-R2
   - mPINNs-R2-2
   - mPINNs-R2-3

2. PPTS-PINNs:
   - NLSE-R1-1
   - NLSE-R1-2
   - NLSE-R1-3
   - NLSE-R2-1
   - NLSE-R2-2
   - NLSE-R2-3

Naming conventions:
- R1 and R2 represent two different types of nonlinear Schrödinger equations (NLSE);
- Suffixes 1, 2, and 3 refer to three noise levels under each equation: 0%, 1%, and 5%;
- R1_data and R2_data are observation datasets corresponding to R1 and R2 equations, respectively.

2. Directory Description for Each Subproject

- data/:
  Contains observational data.

- inverse/:
  - inverse_pfnn*: Defines the PPTS-PINNs neural network model;
  - mPINNs: Defines the mPINNs neural network model;
  - inverse*/mPINNs_inverse*: Implements the complete inversion algorithm.

- *_result/:
  Stores the trained model files.

- model_test:
  Loads and tests the trained models.
