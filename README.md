# nn-derivative-to
Evaluation of Neural Network-based Derivatives for Topology Optimization

By: Joel Najmon and Andres Tovar

This is a Python code repository for the 2023 journal paper: "Evaluation of Neural Network-based Derivatives for Topology Optimization" submitted to ASME's Journal of Mechanical Design in July 2023. The paper is currently under review. This repository includes the multilayer preceptron (MLP) models utilized in the paper and their training scripts, the density-based topology optimization (DBTO) and homogenization-based topology optimization (HBTO) methods, and a general implementation of neural network-based derivative methods for an arbitrary MLP architecture. The four derivative methods implemented in this repository are analytical derivatives, the central finite difference method, the complex step method, and automatic differentiation.

The repository includes the following files:
 * NN_derivatives_examples.py: This script provides a general implementation of the four neural network-based derivative methods for several multivariate regression examples.
 * Train_DBTO_NN.py: This script trains the neural network material model for DBTO.
 * Train_HBTO_NN.py: This script trains the neural network material model for HBTO.
 * Run_DBTO_NN.py: This script executes DBTO on the MBB beam example using the neural network material model.
 * Run_DBTO_SIMP.py: This script executes DBTO on the MBB beam example using the SIMP material model.
 * Run_HBTO_NN.py: This script executes HBTO on the MBB beam example using the neural network material model.
 * Appendix_B_example.py: This script trains the MLP and performs the analytical derivative calculations for the Appendix B example.

 The repository includes the following folders:
 * venv: This folder contains the virtual Python 3.9 environment along with all of the required packages.
 * NN_model_DBTO_1e+04: This folder contains the neural network material model for DBTO trained with TensorFlow.
 * Train_HBTO_NN.py: This script trains the neural network material model for HBTO.
 * Run_DBTO_NN.py: This script executes DBTO on the MBB beam example using the neural network material model.
 * Run_DBTO_SIMP.py: This script executes DBTO on the MBB beam example using the SIMP material model.
 * Run_HBTO_NN.py: This script executes HBTO on the MBB beam example using the neural network material model.
 * Appendix_B_example.py: This script trains the MLP and performs the analytical derivative calculations for the Appendix B example.

\
More detailed instructions are coming soon...




Abstract of the corresponding paper:
\
Neural networks have rapidly grown in popularity for modeling complex non-linear relationships. The computational efficiency and flexibility of neural networks have made them popular for optimization problems, including topology optimization. However, the derivatives of a neural network’s output are crucial for gradient-based optimization algorithms. Recently, there have been several contributions towards improving derivatives of neural network targets; however, there is yet to be a comparative study on the different derivative methods for the sensitivity of the input features on the neural network targets. Therefore, this paper aims to evaluate four derivative methods: analytical derivatives, central finite difference method, complex step method, and automatic differentiation. These methods are implemented into density-based and homogenization-based topology optimization. The derivative methods studied include. For density-based topology optimization, a multilayer perceptron approximates Young's modulus for the solid-isotropic-material-with-penalization (SIMP) model. For homogenization-based topology optimization, a multilayer perceptron approximates the homogenized stiffness tensor of a square cell microstructure with a rectangular hole. The comparative study is performed by solving a two-dimensional topology optimization problem using the sensitivity coefficients from each derivative method. Evaluation includes initial sensitivity coefficients, convergence plots, and the final topologies, compliance, and design variables. The findings demonstrate that neural network-based sensitivity coefficients are sufficient for density-based and homogenization-based topology optimization. The analytical derivative, complex step, and automatic differentiation methods produced identical sensitivity coefficients. The study’s open-source code is provided through an included Python repository.
