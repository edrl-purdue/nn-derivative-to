# nn-derivative-to
Evaluation of Neural Network-based Derivatives for Topology Optimization

By: Joel Najmon and Andres Tovar

This is a Python code repository for the 2023 journal paper: "Evaluation of Neural Network-based Derivatives for Topology Optimization" submitted to ASME's Journal of Mechanical Design in July 2023. The paper is currently under review.



Only lines 7-23 of the MAIN_v3_0_ML_MSTO_Optimizer.m file need to be modified to run the examples. If you wish to skip the computationally expensive de-homogenization step then change line 30 from 'macro.dehom = 1;' to 'macro.dehom = 0;'. Results from the program are found in the 'Results' folder.

Note that there is a typo in the manuscript regarding the delta value for the 3D density-graded and infilled design cases. The delta value for these 3D design cases is delta = 0.9 (i.e., it is not the same delta value of delta = 0.5 used in the 2D density-graded and infilled design cases). This correction is reflected in the 3D example instructions below.

Below are instructions on how to run reduced fidelity versions of the 2D and 3D examples found in the submitted manuscript (simply copy and paste the following lines over lines 7-23).
