/*
 * Using -I/Applications/MuJoCo.app/Contents/Frameworks/mujoco.framework/Versions/A/Headers didn't work for some reason, nothing past mujoco.h was included.
 * I suspect it is because MuJoCo includes the other files with angle brackets <> so they need to be included as system files.
 * Using -isystem to include them didn't fix it though
 * Only thing that worked was copying/linking the files to /usr/local/include and then:
 * clang++  --std=c++17  mjspec_test.cpp -o mjspec_test -lmujoco.3.2.2 -Wall -Wpedantic
 * clang++ --std=c++17 -L /usr/local/lib -l libmujoco.3.2.2 -o mjspec_test mjspec_test.cpp -I /usr/local/include/mujoco
 * install_name_tool -change @rpath/mujoco.framework/Versions/A/libmujoco.3.2.2.dylib /usr/local/lib/libmujoco.3.2.3.dylib mjspec_test
 */

#include <mujoco/mujoco.h>
#include <GLFW/glfw3.h>
#include <iostream>

int main() {
    char error[1000];

    mjSpec* world_spec = mj_parseXML("/Users/Eugene/Developer/robotic_manipulation/mujoco_menagerie/ufactory_lite6/scene.xml", NULL, error, 1000);
    // mjModel* model = mj_loadXML("/Users/Eugene/Developer/robotic_manipulation/mujoco_menagerie/ufactory_lite6/scene.xml", NULL, error, 1000); 
    if (!world_spec) {
        std::cerr << "Failed to load model: " << error << ":(" << std::endl;
        return 1;
    }
    mjSpec* gripper_spec = mj_parseXML("/Users/Eugene/Developer/robotic_manipulation/mujoco_menagerie/ufactory_lite6/gripper_narrow.xml", NULL, error, 1000);
    if (!gripper_spec) {
        std::cerr << "Failed to load model: " << error << ":(" << std::endl;
        return 1;
    }
    // mjsBody* attachment_site = mjs_findBody(world_spec, "link6");
    mjsFrame* attachment_site = mjs_findFrame(world_spec, "attachment_site");
    mjsBody* gripper = mjs_findBody(gripper_spec, "gripper_body");
    mjs_attachBody(attachment_site, gripper, /*prefix=*/"attached-", /*suffix=*/"-1");
    // attachment_site->
    mjModel* m_attached = mj_compile(world_spec, 0);

    return 0; // Success
}