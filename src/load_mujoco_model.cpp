#include "mujoco.h"
#include <iostream>

int main() {
    char error[1000];

    const char* modelPath = "lite_gripper.urdf.xacro"; // Replace with your path
    mjModel* model = mj_loadXML(modelPath, NULL, error, 1000); 

    if (!model) {
        std::cerr << "Failed to load model: " << error << ":(" << std::endl;
        return 1;
    }
    const std::string savePath = "lite6_gripper.xml"; // Where to save
    mj_saveXML(model, savePath.c_str(), error, 1000); 

    return 0; // Success
}