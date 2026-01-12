import numpy as np
import pybullet as p
import pybullet_data
import time
from dex_retargeting.retargeting_config import RetargetingConfig
from retargeting.dex_retarget_controller_mp import DexRukav2Handler
from ruka_hand.control.hand import Hand
from ruka_hand.utils.trajectory import move_to_pos

controller = DexRukav2Handler()

def replay_motion(controller, trajectory_path="mp_trajectory.npy"):
    try:
        data = np.load(trajectory_path)
        print(f"Loaded {data.shape[0]} frames from {trajectory_path}")
    except FileNotFoundError:
        print("Error: Trajectory file not found.")
        return
    first_pos = True

    try:
        for i in range(data.shape[0]):
            points = data[i]
        
            command = controller.get_command(points)
            command[15] = 2400
            command[14] = 1990
            
            curr_pos = controller.hand.read_pos()
            
            if first_pos:
                move_to_pos(curr_pos, command, controller.hand, traj_len=35)
                first_pos = False
            else: 
                move_to_pos(curr_pos, command, controller.hand, traj_len=10)
                # time.sleep(0.5)

        print("Replay finished.")

    except KeyboardInterrupt:
        print("\nReplay interrupted by user.")

if __name__ == "__main__":
    # Assuming 'controller' and 'move_to_pos' are already defined in your environment
    replay_motion(controller)