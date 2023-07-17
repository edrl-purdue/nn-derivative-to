# nn-derivative-to
Evaluation of Neural Network-based Derivatives for Topology Optimization

By: Joel Najmon and Andres Tovar

This is a Python code repository for the 2023 journal paper: "Evaluation of Neural Network-based Derivatives for Topology Optimization" submitted to ASME's Journal of Mechanical Design in July 2023. The paper is currently under review.

The repository includes the following files:
 * NN_derivatives_examples.py: This script provides a general implementation of the four neural network-based derivative methods (i.e., analytical derivative, central finite difference method, complex step method, and automatic differentiation) for several multivariate regression examples.
 * Train_DBTO_NN.py: This script trains the neural network material model for DBTO.
 * Train_HBTO_NN.py: This script trains the neural network material model for HBTO.
 * Run_DBTO_NN.py: This script executes DBTO on the MBB beam example using the neural network material model.
 * Run_DBTO_SIMP.py: This script executes DBTO on the MBB beam example using the SIMP material model.
 * Run_HBTO_NN.py: This script executes HBTO on the MBB beam example using the neural network material model.
 * Appendix_B_example.py: This script trains the MLP and performs the analytical derivative calculations for the Appendix B example.

\
\
More detailed instructions are coming soon...
