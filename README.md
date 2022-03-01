## Probabilistic Adaptive Control ROS package
This ROS package provides an easy-to-integrate C++ implementation of the PAC method for realtime robot control [1][2]. These videos show the robot behavior when controlled through this software at a control rate of 1 kHz:
* https://www.youtube.com/watch?v=OHeAA5slAww&t=3s
* https://www.youtube.com/watch?v=bqWNzkSiFl0

### Usage
pac_node.cpp gives an example of how to initialize and use the PAC class defined in probabilistic_adaptive_control.h. A task is defined by a joint probability distribution of basis function weights and task context descriptors (see Task.msg for details).

### References
[1] J. Jankowski, H. Girgin and S. Calinon, "Probabilistic Adaptive Control for Robust Behavior Imitation," IEEE Robotics and Automation Letters\
[2] J. Jankowski, M. Racca, and S. Calinon, “From Key Positions to Optimal Basis Functions for Probabilistic Adaptive Control,” IEEE Robotics and Automation Letters
