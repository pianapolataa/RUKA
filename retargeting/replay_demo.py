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
    init_pos = [3559, 2864, 2247, 1891, 3407, 849, 3098, 1641, 1490, 853, 230, 1500, 3455, 2860, 2000, 1850]
    wallet_pos = [3559, 3064, 2247, 1791, 3407, 849, 2300, 1641, 600, 853, 230, 800, 2576, 2150, 2000, 1850]
    card_pos = [3559, 3064, 2247, 2000, 3407, 849, 2300, 1641, 600, 853, 230, 800, 2576, 2150, 2000, 1850]

    # wallet_pos = [3559, 3064, 2247, 1791, 3407, 849, 2700, 1641, 400, 853, 230, 800, 2576, 2860, 2000, 1850]
    # card_pos = [3559, 3064, 2247, 2000, 3407, 849, 2700, 1641, 400, 853, 230, 800, 2576, 2860, 2000, 1850]

    current_pos = hand.read_pos()
    move_to_pos(curr_pos=current_pos, des_pos=init_pos, hand=hand, traj_len=35)
    input()
    current_pos = hand.read_pos()
    move_to_pos(curr_pos=current_pos, des_pos=wallet_pos, hand=hand, traj_len=35)
    input()
    current_pos = hand.read_pos()
    move_to_pos(curr_pos=current_pos, des_pos=card_pos, hand=hand, traj_len=35)
    input()

    current_pos = hand.read_pos()
    move_to_pos(curr_pos=current_pos, des_pos=wallet_pos, hand=hand, traj_len=35)
    input()

    current_pos = hand.read_pos()
    move_to_pos(curr_pos=current_pos, des_pos=init_pos, hand=hand, traj_len=35)
    input()

if __name__ == "__main__":
    main()


# import numpy as np
# import time
# from ruka_hand.control.hand import Hand
# from ruka_hand.utils.trajectory import move_to_pos


# def main():
#     hand = Hand("right")
#     front_pos = [2815, 2864, 1384, 1891, 2857, 1824, 1878, 1676, 555, 1947, 1890, 800, 2676, 2257, 2000, 1850]
#     back_pos = [2815, 2864, 1384, 1891, 2857, 1824, 1878, 1676, 555, 1947, 1890, 800, 2676, 2257, 2000, 1200]
#     init_pos = [3559, 2864, 2247, 1891, 3407, 849, 3098, 1641, 1490, 853, 230, 1500, 3455, 2860, 2000, 1850]

#     ##KNOCKING
#     curr_pos = hand.read_pos()
#     move_to_pos(curr_pos=curr_pos, des_pos=init_pos, hand=hand, traj_len=55)

#     for i in range(3):
#         curr_pos = hand.read_pos()
#         move_to_pos(curr_pos=curr_pos, des_pos=front_pos, hand=hand, traj_len=45)
#         curr_pos = hand.read_pos()
#         move_to_pos(curr_pos=curr_pos, des_pos=back_pos, hand=hand, traj_len=45)


#     curr_pos = hand.read_pos()
#     move_to_pos(curr_pos=curr_pos, des_pos=front_pos, hand=hand, traj_len=45)
#     curr_pos = hand.read_pos()
#     move_to_pos(curr_pos=curr_pos, des_pos=init_pos, hand=hand, traj_len=55)

# if __name__ == "__main__":
#     main()


# import cv2
# import numpy as np
# import time
# from dex_retargeting.retargeting_config import RetargetingConfig
# from retargeting.dex_retarget_controller_mp import DexRukav2Handler
# from ruka_hand.control.hand import Hand
# from ruka_hand.utils.trajectory import move_to_pos


# def main():
#     hand = Hand("right")
#     back_pos = [3559, 3000, 2247, 1700, 3407, 849, 3098, 1833, 1490, 853, 230, 1500, 3455, 2860, 2000, 1200]
#     front_pos = [3559, 3000, 2247, 1700, 3407, 849, 3098, 1833, 1490, 853, 230, 1500, 3455, 2860, 2000, 1870]
#     left_pos = [3559, 3000, 2247, 1700, 3407, 849, 3098, 1833, 1490, 853, 230, 1500, 3455, 2860, 2417, 1850]
#     right_pos = [3559, 3000, 2247, 1700, 3407, 849, 3098, 1833, 1490, 853, 230, 1500, 3455, 2860, 1617, 1850]

#     ##WRIST MOTIONS


#     curr_pos = hand.read_pos()
#     move_to_pos(curr_pos=curr_pos, des_pos=front_pos, hand=hand, traj_len=55)

#     for i in range(5):
#         curr_pos = hand.read_pos()
#         move_to_pos(curr_pos=curr_pos, des_pos=left_pos, hand=hand, traj_len=55)
#         curr_pos = hand.read_pos()
#         move_to_pos(curr_pos=curr_pos, des_pos=right_pos, hand=hand, traj_len=55)
    
#     curr_pos = hand.read_pos()
#     move_to_pos(curr_pos=curr_pos, des_pos=front_pos, hand=hand, traj_len=55)

#     for i in range(5):
#         curr_pos = hand.read_pos()
#         move_to_pos(curr_pos=curr_pos, des_pos=front_pos, hand=hand, traj_len=55)
#         time.sleep(0.3)
#         curr_pos = hand.read_pos()
#         move_to_pos(curr_pos=curr_pos, des_pos=back_pos, hand=hand, traj_len=55)
    

#     curr_pos = hand.read_pos()
#     move_to_pos(curr_pos=curr_pos, des_pos=front_pos, hand=hand, traj_len=55)

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

#     init_pos = [3559, 2864, 2247, 1891, 3407, 849, 3098, 1641, 1490, 853, 230, 1500, 3455, 2860, 2000, 1850]
#     all_pos = [3559, 2524, 2247, 2219, 3407, 849, 3098, 1413, 1490, 853, 230, 1500, 3455, 2860, 2000, 1850]

#     pinky_pos = [3559, 2624, 2247, 1891, 3407, 849, 3098, 1641, 1490, 853, 230, 1500, 3455, 2860, 2000, 1850]
#     ring_pos = [3559, 2864, 2247, 2219, 3407, 849, 3098, 1641, 1490, 853, 230, 1500, 3455, 2860, 2000, 1850]
#     index_pos = [3559, 2864, 2247, 1891, 3407, 849, 3098, 1413, 1490, 853, 230, 1500, 3455, 2860, 2000, 1850]

#     ##ABDUCTION
#     time.sleep(0.4)
#     curr_pos = hand.read_pos()
#     move_to_pos(curr_pos=curr_pos, des_pos=init_pos, hand=hand, traj_len=15)
#     time.sleep(0.4)

#     curr_pos = hand.read_pos()
#     move_to_pos(curr_pos=curr_pos, des_pos=index_pos, hand=hand, traj_len=15) 
#     time.sleep(0.4)
#     curr_pos = hand.read_pos()
#     move_to_pos(curr_pos=curr_pos, des_pos=init_pos, hand=hand, traj_len=15)


#     curr_pos = hand.read_pos()
#     time.sleep(0.4)
#     move_to_pos(curr_pos=curr_pos, des_pos=ring_pos, hand=hand, traj_len=15)
#     time.sleep(0.4)
#     curr_pos = hand.read_pos()
#     move_to_pos(curr_pos=curr_pos, des_pos=init_pos, hand=hand, traj_len=15)


#     curr_pos = hand.read_pos()
#     time.sleep(0.4)
#     move_to_pos(curr_pos=curr_pos, des_pos=pinky_pos, hand=hand, traj_len=15)

#     time.sleep(0.4)
#     curr_pos = hand.read_pos()
#     move_to_pos(curr_pos=curr_pos, des_pos=init_pos, hand=hand, traj_len=15)


#     curr_pos = hand.read_pos()
#     time.sleep(0.4)
#     move_to_pos(curr_pos=curr_pos, des_pos=all_pos, hand=hand, traj_len=15)
#     time.sleep(0.4)
#     curr_pos = hand.read_pos()
#     move_to_pos(curr_pos=curr_pos, des_pos=init_pos, hand=hand, traj_len=15)

# if __name__ == "__main__":
#     main()