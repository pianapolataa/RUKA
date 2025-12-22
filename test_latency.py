import argparse
import time
import numpy as np
import cv2

from ruka_hand.control.hand import Hand
from ruka_hand.utils.trajectory import move_to_pos
from get_video_reading import HandReader

parser = argparse.ArgumentParser(description="Teleop robot hands.")
parser.add_argument("-ht", "--hand_type", type=str, default="right")
args = parser.parse_args()
hand = Hand(args.hand_type)

tensioned = hand.tensioned_pos
curled = hand.curled_bound + (hand.tensioned_pos - hand.curled_bound) / 2

sum = 0.0
cnt = 0.0

try:
    while cnt < 100:
        cnt += 1.0        
        curr_pos = hand.read_pos()
        bef = time.perf_counter()
        des_pos = np.random.randint(np.minimum(tensioned, curled), np.maximum(tensioned, curled))
        # print(des_pos)
        move_to_pos(curr_pos=curr_pos, des_pos=des_pos, hand=hand, traj_len=15)
        sum += time.perf_counter() - bef

finally:
    hand.close()
    sum = sum / cnt / 15
    print(sum)

# move_to_pos with traj len 20 sends 20 writes to the motors