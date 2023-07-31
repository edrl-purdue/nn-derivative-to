# nn-derivative-to
# Neural Network-based Derivative Methods for Density-based and Homogenization-based Topology Optimization

### By: Joel Najmon and Andres Tovar

This is a Python code repository for the 2023 journal paper: ***"Evaluation of Neural Network-based Derivatives for Topology Optimization"*** submitted to ASME's Journal of Mechanical Design in July 2023. The paper is currently under review. This repository includes the multilayer preceptron (MLP) models utilized in the paper and their training scripts, the density-based topology optimization (DBTO) and homogenization-based topology optimization (HBTO) methods, and a general implementation of neural network-based derivative methods for an arbitrary MLP architecture. The four derivative methods implemented in this repository are analytical derivatives, the central finite difference method, the complex step method, and automatic differentiation. The repository will be updated as per review.

### The repository includes the following files:
 * **NN_derivatives_examples.py**: This script provides a general implementation of the four neural network-based derivative methods for several multivariate regression examples.
 * **Train_DBTO_NN.py**: This script trains the neural network material model for DBTO.
 * **Train_HBTO_NN.py**: This script trains the neural network material model for HBTO.
 * **Run_DBTO_NN.py**: This script executes DBTO on the MBB beam example using the neural network material model.
 * **Run_DBTO_SIMP.py**: This script executes DBTO on the MBB beam example using the SIMP material model.
 * **Run_HBTO_NN.py**: This script executes HBTO on the MBB beam example using the neural network material model.
 * **Appendix_B_example.py**: This script trains the MLP and performs the analytical derivative calculations for the Appendix B example.
 * **HBTO_CH_2D_N10000_100x100.mat**: A mat database file that contains the parameters and homogenized stiffness tensors of microstructures with a rectangular hole. Used in *Train_HBTO_NN.py*.

### The repository includes the following folders:
 * **NN_model_DBTO_1e+04**: This folder contains the neural network material model for DBTO trained with TensorFlow.
 * **NN_model_HBTO_1e+04**: This folder contains the neural network material model for HBTO trained with TensorFlow.
 * **JMD Journal Paper Results**: This folder contains results from the DBTO and HBTO scripts that were presented in the paper.
 * **DBTO_NN_result_dy1**: This folder contains the outputs from *Run_DBTO_NN.py* when using neural network-based sensitivity coefficients produced with analytical derivatives (i.e., dy1).
 * **DBTO_NN_result_dy2**: This folder contains the outputs from *Run_DBTO_NN.py* when using neural network-based sensitivity coefficients produced with the central finite difference method (i.e., dy2).
 * **DBTO_NN_result_dy3**: This folder contains the outputs from *Run_DBTO_NN.py* when using neural network-based sensitivity coefficients produced with the complex step method (i.e., dy3).
 * **DBTO_NN_result_dy4**: This folder contains the outputs from *Run_DBTO_NN.py* when using neural network-based sensitivity coefficients produced with automatic differentiation (i.e., dy4).
 * **DBTO_SIMP_result**: This folder contains the outputs from *Run_DBTO_SIMP.py* when using sensitivity coefficients analytically derived from the SIMP material model.
 * **HBTO_NN_result_dy1**: This folder contains the outputs from *Run_HBTO_NN.py* when using neural network-based sensitivity coefficients produced with analytical derivatives (i.e., dy1).
 * **HBTO_NN_result_dy2**: This folder contains the outputs from *Run_HBTO_NN.py* when using neural network-based sensitivity coefficients produced with the central finite difference method (i.e., dy2).
 * **HBTO_NN_result_dy3**: This folder contains the outputs from *Run_HBTO_NN.py* when using neural network-based sensitivity coefficients produced with the complex step method (i.e., dy3).
 * **HBTO_NN_result_dy4**: This folder contains the outputs from *Run_HBTO_NN.py* when using neural network-based sensitivity coefficients produced with automatic differentiation (i.e., dy4).

### Instructions for running examples:
**NN_derivatives_examples.py**
 * Run this script for a simple example of neural network-derived derivatives.
 * Lines 14-179 provide several multivariable regression functions as examples. The input and output formatting is detailed here for user modification.
 * Line 182 defines the total number of features sets. The default dataset division is 70% for the training dataset, and 15% for the testing and validation datasets, each.
 * Lines 192-195 define the number of hidden layers, number of neurons per hidden layer, the hidden activation function, and the output activation function.
 * New activation functions and their derivatives can be added to the functions of Lines 225 and 235.
 * The NN's performance and the error of the four derivative methods is displayed.
 * For low dimensional example functions, the function and its derivative are plotted for the ground-truth and neural network-based evaluations.

**Train_DBTO_NN.py**
 * Run this script to train a neural network material model for DBTO similar to the one utilized in the paper. Note that this script is not deterministic.
 * Line 30 defines the total number of features sets. The default dataset division is 70% for the training dataset, and 15% for the testing and validation datasets, each.
 * Lines 41-45 define the number of hidden layers, number of neurons per hidden layer, the hidden activation function, the output activation function, and the number of repeated NNs to train when searching for the best peforming one.
 * The NN's performance is displayed along with plots of the function and its derivative from the ground-truth function and the neural network-based evaluation.
 * The model is saved in the NN_model_DBTO_## folder where ## is the total number of features sets defined on Line 30.

 **Train_HBTO_NN.py**
 * Run this script to train a neural network material model for HBTO similar to the one utilized in the paper. Note that this script is not deterministic.
 * Line 14 loads the precomputed database file: HBTO_CH_2D_N10000_100x100.mat. The default dataset division is 70% for the training dataset, and 15% for the testing and validation datasets, each.
 * Lines 32-36 define the number of hidden layers, number of neurons per hidden layer, the hidden activation function, the output activation function, and the number of repeated NNs to train when searching for the best peforming one.
 * * The NN's performance is displayed along with plots of the function and its derivative from the ground-truth function and the neural network-based evaluation.
 * The model is saved in the NN_model_HBTO_## folder where ## is the total number of features sets loaded on Line 14.

**Run_DBTO_NN.py**
 * Run this script to perform DBTO on an MBB beam example with the neural network material model.
 * Line 489-496 can be modified to adjust the model and optimization settings.
 * Line 503 is where the name of the NN model folder is supplied for loading.
 * Line 504 is where the type of neural network-based derivative method is selected (i.e., 1 = Analytical Der., 2 = CFD, 3 = CSM, 4 = Automatic Diff.).
 * Lines 506-519 are where the loads and boundary conditions of the MBB beam example are defined.
 * The final output is displayed and saved in the DBTO_NN_result_dy# folder.

 **Run_DBTO_SIMP.py**
 * Run this script to perform DBTO on an MBB beam example with the SIMP material model.
 * Line 363-371 can be modified to adjust the model and optimization settings.
 * Lines 379-392 are where the loads and boundary conditions of the MBB beam example are defined.
 * The final output is displayed and saved in the DBTO_SIMP_result folder.

 **Run_HBTO_NN.py**
 * Run this script to perform HBTO on an MBB beam example with the neural network material model.
 * Line 512-519 can be modified to adjust the model and optimization settings.
 * Line 526 is where the name of the NN model folder is supplied for loading.
 * Line 527 is where the type of neural network-based derivative method is selected (i.e., 1 = Analytical Der., 2 = CFD, 3 = CSM, 4 = Automatic Diff.).
 * Lines 529-542 are where the loads and boundary conditions of the MBB beam example are defined.
 * The final output is displayed and saved in the HBTO_NN_result_dy# folder.

 **Appendix_B_example.py**
 * Run this script perform the analytical derivative example of Appendix B. Note that this script is deterministic and will produce the same values found in the appendix.
 * Line 29 defines the total number of features sets. The default dataset division is 70% for the training dataset, and 15% for the testing and validation datasets, each.
 * Lines 40-43 define the number of hidden layers, number of neurons per hidden layer, the hidden activation function, and the output activation function.
 * The NN's performance is displayed along with plots of the function and its derivative from the ground-truth function and the neural network-based evaluation.
 * The first value of the testing dataset is the query input that is demonstrated in the appendix.

### Abstract of the corresponding paper, Evaluation of Neural Network-based Derivatives for Topology Optimization:
Neural networks have rapidly grown in popularity for modeling complex non-linear relationships. The computational efficiency and flexibility of neural networks have made them popular for optimization problems, including topology optimization. However, the derivatives of a neural network’s output are crucial for gradient-based optimization algorithms. Recently, there have been several contributions towards improving derivatives of neural network targets; however, there is yet to be a comparative study on the different derivative methods for the sensitivity of the input features on the neural network targets. Therefore, this paper aims to evaluate four derivative methods: analytical derivatives, central finite difference method, complex step method, and automatic differentiation. These methods are implemented into density-based and homogenization-based topology optimization. The derivative methods studied include. For density-based topology optimization, a multilayer perceptron approximates Young's modulus for the solid-isotropic-material-with-penalization (SIMP) model. For homogenization-based topology optimization, a multilayer perceptron approximates the homogenized stiffness tensor of a square cell microstructure with a rectangular hole. The comparative study is performed by solving a two-dimensional topology optimization problem using the sensitivity coefficients from each derivative method. Evaluation includes initial sensitivity coefficients, convergence plots, and the final topologies, compliance, and design variables. The findings demonstrate that neural network-based sensitivity coefficients are sufficient for density-based and homogenization-based topology optimization. The analytical derivative, complex step, and automatic differentiation methods produced identical sensitivity coefficients. The study’s open-source code is provided through an included Python repository.
