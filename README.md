# nn-derivative-to
Evaluation of Neural Network-based Derivatives for Topology Optimization

By: Joel Najmon and Andres Tovar

This is a Python code repository for the 2023 journal paper: "Evaluation of Neural Network-based Derivatives for Topology Optimization" submitted to ASME's Journal of Mechanical Design in July 2023. The paper is currently under review.

The repository includes the following files:
\
\
NN_derivatives_examples.py: This script provides a general implementation of the four neural network-based derivative methods (i.e., analytical derivative, central finite difference method, complex step method, and automatic differentiation) for several multivariate regression examples.
\
\
Train_DBTO_NN.py: This script trains the neural network material model for DBTO.
\
\
Train_HBTO_NN.py: This script trains the neural network material model for HBTO.
\
\
Run_DBTO_NN.py: This script executes DBTO on the MBB beam example using the neural network material model.
\
\
Run_DBTO_SIMP.py: This script executes DBTO on the MBB beam example using the SIMP material model.
\
\
Run_HBTO_NN.py: This script executes HBTO on the MBB beam example using the neural network material model.
\
\
Appendix_B_example.py: This script trains the MLP and performs the analytical derivative calculations for the Appendix \ref{secB_appendix} example.
\
\
Only lines 7-23 of the MAIN_v3_0_ML_MSTO_Optimizer.m file need to be modified to run the examples. If you wish to skip the computationally expensive de-homogenization step then change line 30 from 'macro.dehom = 1;' to 'macro.dehom = 0;'. Results from the program are found in the 'Results' folder.

Note that there is a typo in the manuscript regarding the delta value for the 3D density-graded and infilled design cases. The delta value for these 3D design cases is delta = 0.9 (i.e., it is not the same delta value of delta = 0.5 used in the 2D density-graded and infilled design cases). This correction is reflected in the 3D example instructions below.

Below are instructions on how to run reduced fidelity versions of the 2D and 3D examples found in the submitted manuscript (simply copy and paste the following lines over lines 7-23).
