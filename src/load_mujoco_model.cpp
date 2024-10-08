/*
 * Using -I/Applications/MuJoCo.app/Contents/Frameworks/mujoco.framework/Versions/A/Headers didn't work for some reason, nothing past mujoco.h was included.
 * I suspect it is because MuJoCo includes the other files with angle brackets <> so they need to be included as system files.
 * Using -isystem to include them didn't fix it though
 * Only thing that worked was copying/linking the files to /usr/local/include and then:
 * clang++ --std=c++17 -o load_mujoco load_mujoco_model.cpp -I /usr/local/include/mujoco -L /usr/local/lib -l mujoco.3.2.2
 */

#include <mujoco/mujoco.h>
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