At the moment, the control part is disabled and the code only performs the target measurement and joint angle measurement. You can see the output of desired joint angles and measured joint angles.
You can change the desired angle in the initializer of the node image2.py.

For the control part, set "self.CONTROL" in image2.py to True (in the initializer).
To get the result, use for example: "rqt_plot /end_effector/x_position_measured /target/x_position_measured /target/x_position_controller/command"


