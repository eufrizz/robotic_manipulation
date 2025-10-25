import numpy as np

class GraspAndLiftTask(object):
    def __init__(self, l_gripper_name, r_gripper_name, box_name, floor_name) -> None:
        """
        geom ids
        """
        self.l_gripper_name = l_gripper_name
        self.r_gripper_name = r_gripper_name
        self.box_name = box_name
        self.floor_name = floor_name
        self.max_reward = 5
        self.task_description = "Grasp and lift red cube"
    
    def get_reward(self, model, data):
        """
        1 - close to box and not touching ground
        2 - one gripper touching box and not touching ground
        3 - two grippers touching box and not touching ground
        4 - two grippers touching box and box not touching ground
        5 - two grippers touching box and box > 0.2 in z
        """

        l_gripper_touching_ground = False
        r_gripper_touching_ground = False
        box_touching_ground = False
        l_gripper_touching_box = False
        r_gripper_touching_box = False

        for geom in data.contact.geom:
            if all(np.isin(geom, [model.geom(self.floor_name).id, model.geom(self.l_gripper_name).id])):
              l_gripper_touching_ground = True
            
            if all(np.isin(geom, [model.geom(self.floor_name).id, model.geom(self.r_gripper_name).id])):
              r_gripper_touching_ground = True
                        
            if all(np.isin(geom, [model.geom(self.floor_name).id, model.geom(self.box_name).id])):
              box_touching_ground = True
            

            if all(np.isin(geom, [model.geom(self.box_name).id, model.geom(self.l_gripper_name).id])):
              l_gripper_touching_box = True
            if all(np.isin(geom, [model.geom(self.box_name).id,  model.geom(self.r_gripper_name).id])):
              r_gripper_touching_box = True
        
        gripper_touching_ground = l_gripper_touching_ground or r_gripper_touching_ground
        dist_to_box = np.linalg.norm(data.body(self.l_gripper_name).xpos - data.body(self.box_name).xpos)
        close_to_box = dist_to_box < 0.1
        box_above_height = data.body(self.box_name).xpos[2] > 0.2

        # print(f"gripper_touching_ground: {gripper_touching_ground}, box_touching_ground: {box_touching_ground}, l_gripper_touching_box: {l_gripper_touching_box}, r_gripper_touching_box: {r_gripper_touching_box}, close_to_box: {close_to_box}, box_above_height: {box_above_height}")

        reward = 0
        if l_gripper_touching_box and r_gripper_touching_box and box_above_height:
            assert(not box_touching_ground)
            assert(not gripper_touching_ground)
            assert(close_to_box)
            reward = 5
        elif l_gripper_touching_box and r_gripper_touching_box and not box_touching_ground and not gripper_touching_ground:
            assert(close_to_box)
            reward = 4
        elif l_gripper_touching_box and r_gripper_touching_box and not gripper_touching_ground:
            assert(close_to_box)
            reward = 3
        elif l_gripper_touching_box or r_gripper_touching_box and not gripper_touching_ground:
            assert(close_to_box)
            reward = 2
        elif close_to_box and not gripper_touching_ground:
            reward = 1 - (dist_to_box / 0.1)
        
        return reward
  

class GraspTask(object):
    def __init__(self, l_gripper_name, r_gripper_name, box_name, floor_name) -> None:
        """
        geom ids
        """
        self.l_gripper_name = l_gripper_name
        self.r_gripper_name = r_gripper_name
        self.box_name = box_name
        self.floor_name = floor_name
        self.max_reward = 3
        self.task_description = "Grasp the red cube"
    
    def get_reward(self, model, data):
        """
        1 - close to box and not touching ground
        2 - one gripper touching box and not touching ground
        3 - two grippers touching box and not touching ground

        """

        l_gripper_touching_ground = False
        r_gripper_touching_ground = False
        box_touching_ground = False
        l_gripper_touching_box = False
        r_gripper_touching_box = False

        for geom in data.contact.geom:
            if all(np.isin(geom, [model.geom(self.floor_name).id, model.geom(self.l_gripper_name).id])):
              l_gripper_touching_ground = True
            
            if all(np.isin(geom, [model.geom(self.floor_name).id, model.geom(self.r_gripper_name).id])):
              r_gripper_touching_ground = True            

            if all(np.isin(geom, [model.geom(self.box_name).id, model.geom(self.l_gripper_name).id])):
              l_gripper_touching_box = True
            if all(np.isin(geom, [model.geom(self.box_name).id,  model.geom(self.r_gripper_name).id])):
              r_gripper_touching_box = True
        
        gripper_touching_ground = l_gripper_touching_ground or r_gripper_touching_ground
        dist_to_box = np.linalg.norm(data.body(self.l_gripper_name).xpos - data.body(self.box_name).xpos)
        close_to_box = dist_to_box < 0.1

        # print(f"gripper_touching_ground: {gripper_touching_ground}, box_touching_ground: {box_touching_ground}, l_gripper_touching_box: {l_gripper_touching_box}, r_gripper_touching_box: {r_gripper_touching_box}, close_to_box: {close_to_box}, box_above_height: {box_above_height}")

        reward = 0
        if l_gripper_touching_box and r_gripper_touching_box and not gripper_touching_ground:
            assert(close_to_box)
            reward = 3
        elif l_gripper_touching_box or r_gripper_touching_box and not gripper_touching_ground:
            assert(close_to_box)
            reward = 2
        elif close_to_box and not gripper_touching_ground:
            reward = 1 - (dist_to_box / 0.1)
        
        return reward
  

