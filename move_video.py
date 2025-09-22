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

# # Min/max degrees for each motor
min_deg = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -25, 0], dtype=float)
max_deg = np.array([90, 40, 85, 15, 90, 85, 70, 20, 90, 80, 90, 90, 145, 90, 25, 50], dtype=float)

def normalize_to_motor(test_pos):
    test_pos = np.array(test_pos, dtype=float)
    print(test_pos)
    clamped = np.clip(test_pos, min_deg, max_deg)
    normed = clamped / (max_deg - min_deg)
    positions = normed * (hand.curled_bound - hand.tensioned_pos) + hand.tensioned_pos
    positions[1] = 2285 + normed[1] * abs(hand.curled_bound[1] - hand.tensioned_pos[1])
    positions[3] = 2070 - normed[3] * abs(hand.curled_bound[3] - hand.tensioned_pos[3])
    positions[7] = 2125 + normed[7] * abs(hand.curled_bound[7] - hand.tensioned_pos[7])
    positions[14] = 1990 + normed[14] * abs(hand.curled_bound[14] - hand.tensioned_pos[14])
    return positions

reader = HandReader()

try:
    while True:
        motor_positions, frame = reader.get_motor_positions()
        if motor_positions is not None:
            try:
                curr_pos = hand.read_pos()
                print("got reading")
                des_pos = normalize_to_motor(motor_positions)
                print(des_pos)
                # input()
                move_to_pos(curr_pos=curr_pos, des_pos=des_pos, hand=hand, traj_len=40)
            except Exception as e:
                print(f"Error: {e}")

        if frame is not None:
            cv2.imshow("Teleop", frame)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
                break
finally:
    reader.release()
    hand.close()
