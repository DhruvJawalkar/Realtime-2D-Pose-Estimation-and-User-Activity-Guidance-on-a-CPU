import math
import numpy as np

def get_angle(pt_pair1, pt_pair2):
    ax,ay = pt_pair1[1][0] - pt_pair1[0][0], pt_pair1[1][1] - pt_pair1[0][1]
    bx,by = pt_pair2[1][0] - pt_pair2[0][0], pt_pair2[1][1] - pt_pair2[0][1]
    mod_a = math.sqrt(ax**2 + ay**2)
    mod_b = math.sqrt(bx**2 + by**2)
    angle = math.degrees(math.acos((ax*bx + ay*by) / ((mod_a*mod_b) or 1) ))
    if(angle==math.nan):
        angle=-1 
    return angle

def check_if_feet_are_together(kps):
    threshold = 60
    if(np.abs(np.sum(kps[10][0] - kps[13][0]))<=threshold):
        return True
    return False    

def check_if_arms_on_hip(kps):
    angle1 = get_angle((kps[2], kps[3]), (kps[3], kps[4]))
    angle2 = get_angle((kps[5], kps[6]), (kps[6], kps[7]))

    if((angle1>= 50 and angle1 <=90) or (angle2>= 50 and angle2 <=90)):
        return True
    return False

def check_if_feet_are_apart(kps):
    angle = get_angle((kps[9], kps[10]), (kps[12], kps[13]))
    if(angle >35 and angle <=60):
        return True
    return False

def check_hip_twist(kps):
    threshold = 70
    if(np.abs(np.sum(kps[8] - kps[11]))<=threshold):
        return True
    return False

def check_arm_raise(kps):
    angle1 = get_angle((kps[2], kps[8]), (kps[2], kps[3]))
    angle2 = get_angle((kps[5], kps[11]), (kps[5], kps[6]))

    if((130<=angle1 and angle1<=180) or (130<=angle2 and angle2<=180)):
        return True
    return False

def check_lunge_angle(kps):
    angle = get_angle((kps[8], kps[9]), (kps[9], kps[10]))

    if(60<=angle and angle<=95):
        return True
    return False

def check_back_leg_is_straight(kps):
    angle = get_angle((kps[11], kps[12]), (kps[12], kps[13]))
    
    if(0<=angle and angle<=15):
        return True
    return False