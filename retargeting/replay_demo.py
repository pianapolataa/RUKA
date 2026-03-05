# import cv2
# import mediapipe as mp
# import numpy as np
# import pybullet as p
# import pybullet_data
# import time
# from dex_retargeting.retargeting_config import RetargetingConfig
# from retargeting.dex_retarget_controller_mp import DexRukav2Handler
# from ruka_hand.control.hand import Hand
# from ruka_hand.utils.trajectory import move_to_pos


# def main():
#     hand = Hand("right")
#     init_pos = [2400, 2540, 2550, 1860, 2320, 1920, 2500, 2100, 2200, 2300, 2550, 1540, 1400, 2080, 1990, 1779]
#     wallet_pos = [2400, 2540, 2550, 1860, 2320, 1920, 1520, 2100, 1573, 2300, 2550, 1470, 720, 1270, 1990, 1779]
#     card_pos = [2400, 2540, 2550, 2160, 2320, 1920, 1520, 2100, 1673, 2300, 2550, 1470, 720, 1270, 1990, 1779]
#     keys_pos = [2400, 2540, 1617, 2160, 2320, 1920, 1520, 2100, 1673, 2300, 2550, 1470, 720, 1270, 1990, 1779]

#     current_pos = hand.read_pos()
#     move_to_pos(curr_pos=current_pos, des_pos=init_pos, hand=hand, traj_len=35)
#     input()
#     current_pos = hand.read_pos()
#     move_to_pos(curr_pos=current_pos, des_pos=wallet_pos, hand=hand, traj_len=35)
#     input()
#     current_pos = hand.read_pos()
#     move_to_pos(curr_pos=current_pos, des_pos=card_pos, hand=hand, traj_len=35)
#     input()
#     current_pos = hand.read_pos()
#     move_to_pos(curr_pos=current_pos, des_pos=keys_pos, hand=hand, traj_len=35)

# if __name__ == "__main__":
#     main()


# import cv2
# import mediapipe as mp
# import numpy as np
# import pybullet as p
# import pybullet_data
# import time
# from dex_retargeting.retargeting_config import RetargetingConfig
# from retargeting.dex_retarget_controller_mp import DexRukav2Handler
# from ruka_hand.control.hand import Hand
# from ruka_hand.utils.trajectory import move_to_pos


# def main():
#     hand = Hand("right")
#     front_pos = [1850, 2540, 1617, 1860, 1730, 2900, 1520, 1850, 1573, 3229, 3237, 1300, 520, 1430, 1990, 1779]
#     back_pos = [1850, 2540, 1617, 1860, 1730, 2900, 1520, 1850, 1573, 3229, 3237, 1300, 520, 1430, 1990, 2455]

#     ##KNOCKING

#     while True:
#         curr_pos = hand.read_pos()
#         move_to_pos(curr_pos=curr_pos, des_pos=front_pos, hand=hand, traj_len=55)
#         curr_pos = hand.read_pos()
#         move_to_pos(curr_pos=curr_pos, des_pos=back_pos, hand=hand, traj_len=55)

# if __name__ == "__main__":
#     main()


# import cv2
# import mediapipe as mp
# import numpy as np
# import pybullet as p
# import pybullet_data
# import time
# from dex_retargeting.retargeting_config import RetargetingConfig
# from retargeting.dex_retarget_controller_mp import DexRukav2Handler
# from ruka_hand.control.hand import Hand
# from ruka_hand.utils.trajectory import move_to_pos


# def main():
#     hand = Hand("right")
#     back_pos = [2400, 2540, 2550, 1860, 2320, 1920, 2500, 2100, 2200, 2300, 2550, 1540, 1400, 2080, 1990, 1770]
#     front_pos = [2400, 2540, 2550, 1860, 2320, 1920, 2500, 2100, 2200, 2300, 2550, 1540, 1400, 2080, 1990, 2455]
#     left_pos = [2400, 2540, 2550, 1860, 2320, 1920, 2500, 2100, 2200, 2300, 2550, 1540, 1400, 2080, 1617, 2455]
#     right_pos = [2400, 2540, 2550, 1860, 2320, 1920, 2500, 2100, 2200, 2300, 2550, 1540, 1400, 2080, 2417, 2455]

#     ##WRIST MOTIONS

#     while True:
#         # curr_pos = hand.read_pos()
#         # move_to_pos(curr_pos=curr_pos, des_pos=left_pos, hand=hand, traj_len=75)
#         # curr_pos = hand.read_pos()
#         # move_to_pos(curr_pos=curr_pos, des_pos=right_pos, hand=hand, traj_len=75)


#         curr_pos = hand.read_pos()
#         move_to_pos(curr_pos=curr_pos, des_pos=front_pos, hand=hand, traj_len=75)
#         curr_pos = hand.read_pos()
#         move_to_pos(curr_pos=curr_pos, des_pos=back_pos, hand=hand, traj_len=75)

# if __name__ == "__main__":
#     main()


import cv2
import mediapipe as mp
import numpy as np
import pybullet as p
import pybullet_data
import time
from dex_retargeting.retargeting_config import RetargetingConfig
from retargeting.dex_retarget_controller_mp import DexRukav2Handler
from ruka_hand.control.hand import Hand
from ruka_hand.utils.trajectory import move_to_pos


def main():
    hand = Hand("right")

    init_pos = [2400, 2140, 2550, 2160, 2320, 1920, 2500, 1850, 2200, 2300, 2550, 1540, 1400, 2080, 1990, 2455]
    ring_pos = [2400, 2540, 2550, 1860, 2320, 1920, 2500, 1850, 2200, 2300, 2550, 1540, 1400, 2080, 1990, 2455]
    pinky_pos = [2400, 2540, 2550, 2160, 2320, 1920, 2500, 1850, 2200, 2300, 2550, 1540, 1400, 2080, 1990, 2455]
    final_pos = [2400, 2540, 2550, 1860, 2320, 1920, 2500, 2100, 2200, 2300, 2550, 1540, 1400, 2080, 1990, 2455]

    ##ABDUCTION
    curr_pos = hand.read_pos()
    move_to_pos(curr_pos=curr_pos, des_pos=init_pos, hand=hand, traj_len=20)
    curr_pos = hand.read_pos()
    time.sleep(1)
    move_to_pos(curr_pos=curr_pos, des_pos=pinky_pos, hand=hand, traj_len=20)
    curr_pos = hand.read_pos()
    time.sleep(1)
    move_to_pos(curr_pos=curr_pos, des_pos=ring_pos, hand=hand, traj_len=20)
    curr_pos = hand.read_pos()
    time.sleep(1)
    move_to_pos(curr_pos=curr_pos, des_pos=final_pos, hand=hand, traj_len=20)

if __name__ == "__main__":
    main()