# nn-derivative-to
# Neural Network-based Derivative Methods for Density-based and Homogenization-based Topology Optimization

### By: Joel Najmon and Andres Tovar

This is a Python code repository for the journal paper: ***"Evaluation of Neural Network-based Derivatives for Topology Optimization"*** accepted by ASME's Journal of Mechanical Design in November 2023. This repository includes the multilayer perceptron (MLP) models utilized in the paper and their training scripts, the density-based topology optimization (DBTO) and homogenization-based topology optimization (HBTO) methods, and a general implementation of neural network-based derivative methods for arbitrary MLP architectures. The four derivative methods implemented in this repository are analytical derivatives through the neural network (referred to as neural network Jacobian or NNJ), the central finite difference (CFD) method, the complex step method (CSM), and automatic differentiation (AD). Additionally, a preprint of the accepted manuscript may be found in the repository. The authors request that users of this repository cite the accompanying JMD journal paper.

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
 * **DBTO_NN_result_dy1**: This folder contains the outputs from *Run_DBTO_NN.py* when using neural network-based sensitivity coefficients produced with NNJ (i.e., dy1).
 * **DBTO_NN_result_dy2**: This folder contains the outputs from *Run_DBTO_NN.py* when using neural network-based sensitivity coefficients produced with CFD (i.e., dy2).
 * **DBTO_NN_result_dy3**: This folder contains the outputs from *Run_DBTO_NN.py* when using neural network-based sensitivity coefficients produced with CSM (i.e., dy3).
 * **DBTO_NN_result_dy4**: This folder contains the outputs from *Run_DBTO_NN.py* when using neural network-based sensitivity coefficients produced with AD (i.e., dy4).
 * **DBTO_SIMP_result**: This folder contains the outputs from *Run_DBTO_SIMP.py* when using sensitivity coefficients analytically derived from the SIMP material model.
 * **HBTO_NN_result_dy1**: This folder contains the outputs from *Run_HBTO_NN.py* when using neural network-based sensitivity coefficients produced with NNJ (i.e., dy1).
 * **HBTO_NN_result_dy2**: This folder contains the outputs from *Run_HBTO_NN.py* when using neural network-based sensitivity coefficients produced with the CFD (i.e., dy2).
 * **HBTO_NN_result_dy3**: This folder contains the outputs from *Run_HBTO_NN.py* when using neural network-based sensitivity coefficients produced with the CSM (i.e., dy3).
 * **HBTO_NN_result_dy4**: This folder contains the outputs from *Run_HBTO_NN.py* when using neural network-based sensitivity coefficients produced with AD (i.e., dy4).

### Instructions for running examples:
**NN_derivatives_examples.py**
 * Run this script for a simple example of neural network-derived derivatives.
 * Lines 14-179 provide several multivariable regression functions as examples. The input and output formatting is detailed here for user modification.
 * Line 182 defines the total number of feature sets. The default dataset division is 70% for the training dataset, and 15% for the testing and validation datasets, each.
 * Lines 192-195 define the number of hidden layers, the number of neurons per hidden layer, the hidden activation function, and the output activation function.
 * New activation functions and their derivatives can be added to the functions of Lines 225 and 235.
 * The NN's performance and the error of the four derivative methods are displayed.
 * For low dimensional example functions, the function and its derivative are plotted for the ground-truth and neural network-based evaluations.

**Train_DBTO_NN.py**
 * Run this script to train a neural network material model for DBTO similar to the one utilized in the paper. Note that this script is not deterministic.
 * Line 30 defines the total number of feature sets. The default dataset division is 70% for the training dataset, and 15% for the testing and validation datasets, each.
 * Lines 41-45 define the number of hidden layers, the number of neurons per hidden layer, the hidden activation function, the output activation function, and the number of repeated NNs to train when searching for the best-performing one.
 * The NN's performance is displayed along with plots of the function and its derivative from the ground-truth function and the neural network-based evaluation.
 * The model is saved in the *NN_model_DBTO_##* folder where ## is the total number of feature sets defined on Line 30.

 **Train_HBTO_NN.py**
 * Run this script to train a neural network material model for HBTO similar to the one utilized in the paper. Note that this script is not deterministic.
 * Line 14 loads the precomputed database file: *HBTO_CH_2D_N10000_100x100.mat*. The default dataset division is 70% for the training dataset, and 15% for the testing and validation datasets, each.
 * Lines 32-36 define the number of hidden layers, the number of neurons per hidden layer, the hidden activation function, the output activation function, and the number of repeated NNs to train when searching for the best-performing one.
 * The NN's performance is displayed along with plots of the function and its derivative from the ground-truth function and the neural network-based evaluation.
 * The model is saved in the *NN_model_HBTO_##* folder where ## is the total number of feature sets loaded on Line 14.

**Run_DBTO_NN.py**
 * Run this script to perform DBTO on an MBB beam example with the neural network material model.
 * Lines 489-496 can be modified to adjust the model and optimization settings.
 * Line 503 is where the name of the NN model folder is supplied for loading.
 * Line 504 is where the type of neural network-based derivative method is selected (i.e., 1 = Analytical Der., 2 = CFD, 3 = CSM, 4 = Automatic Diff.).
 * Lines 506-519 are where the loads and boundary conditions of the MBB beam example are defined.
 * The final output is displayed and saved in the *DBTO_NN_result_dy#* folder.

 **Run_DBTO_SIMP.py**
 * Run this script to perform DBTO on an MBB beam example with the SIMP material model.
 * Lines 363-371 can be modified to adjust the model and optimization settings.
 * Lines 379-392 are where the loads and boundary conditions of the MBB beam example are defined.
 * The final output is displayed and saved in the *DBTO_SIMP_result* folder.

 **Run_HBTO_NN.py**
 * Run this script to perform HBTO on an MBB beam example with the neural network material model.
 * Lines 512-519 can be modified to adjust the model and optimization settings.
 * Line 526 is where the name of the NN model folder is supplied for loading.
 * Line 527 is where the type of neural network-based derivative method is selected (i.e., 1 = Analytical Der., 2 = CFD, 3 = CSM, 4 = Automatic Diff.).
 * Lines 529-542 are where the loads and boundary conditions of the MBB beam example are defined.
 * The final output is displayed and saved in the *HBTO_NN_result_dy#* folder.

 **Appendix_B_example.py**
 * Run this script to perform the analytical derivative example of Appendix B. Note that this script is deterministic and will produce the same values found in the appendix.
 * Line 29 defines the total number of feature sets. The default dataset division is 70% for the training dataset, and 15% for the testing and validation datasets, each.
 * Lines 40-43 define the number of hidden layers, the number of neurons per hidden layer, the hidden activation function, and the output activation function.
 * The NN's performance is displayed along with plots of the function and its derivative from the ground-truth function and the neural network-based evaluation.
 * The first value of the testing dataset is the query input that is demonstrated in the appendix.

### Abstract of the corresponding JMD paper, Evaluation of Neural Network-based Derivatives for Topology Optimization:
Neural networks have gained popularity for modeling complex non-linear relationships. Their computational efficiency has led to their growing adoption in optimization methods, including topology optimization. Recently, there have been several contributions towards improving derivatives of neural network outputs, which can improve their use in gradient-based optimization. However, a comparative study has yet to be conducted on the different derivative methods for the sensitivity of the input features on the neural network outputs. This paper aims to evaluate four derivative methods: analytical neural network's Jacobian, central finite difference method, complex step method, and automatic differentiation. These methods are implemented into density-based and homogenization-based topology optimization using multilayer perceptrons (MLPs). For density-based topology optimization, the MLP approximates Young's modulus for the solid-isotropic-material-with-penalization (SIMP) model. For homogenization-based topology optimization, the MLP approximates the homogenized stiffness tensor of a representative volume element, e.g., square cell microstructure with a rectangular hole. The comparative study is performed by solving two-dimensional topology optimization problems using the sensitivity coefficients from each derivative method. Evaluation includes initial sensitivity coefficients, convergence plots, and the final topologies, compliance, and design variables. The findings demonstrate that neural network-based sensitivity coefficients are sufficiently accurate for density-based and homogenization-based topology optimization. The neural network's Jacobian, complex step method, and automatic differentiation produced identical sensitivity coefficients to working precision. The study’s open-source code is provided through a Python repository.
