import argparse

import cv2
import numpy as np
import torch

from models.with_mobilenet import PoseEstimationWithMobileNet
from modules.keypoints import extract_keypoints, group_keypoints
from modules.load_state import load_state
from modules.pose import Pose, propagate_ids
from val import normalize, pad_width
from demo_utils import State, Activity
from demo_checks import *
from demo_controller import Controller

class PersistMessage:
    def __init__(self):
        self.video_frame_ref = ''
        self.message = ''
        self.image = ''

    def set_message(self, video_frame_ref, msg, img, time):
        global PERSIST_MESSAGE_COUNTER
        global PERSIST_MESSAGE_FLAG
        self.video_frame_ref = video_frame_ref
        self.message = msg
        self.image = img
        if(time):
            PERSIST_MESSAGE_COUNTER = time
            PERSIST_MESSAGE_FLAG = True

    def update_video_frame_ref(self, ref):
        self.video_frame_ref = ref

    def decrement_persistance_counter(self):
        global PERSIST_MESSAGE_COUNTER
        PERSIST_MESSAGE_COUNTER -=1

    def overlay_image(self, video_frame, img_overlay):
        video_frame[0:img_overlay.shape[0], video_frame.shape[1]-img_overlay.shape[1]:video_frame.shape[1]] = img_overlay    

    def persist(self):
        global PERSIST_MESSAGE_COUNTER
        global PERSIST_MESSAGE_FLAG
        if(PERSIST_MESSAGE_COUNTER>0):
            cv2.putText(self.video_frame_ref, self.message, (20,20), cv2.FONT_HERSHEY_COMPLEX, 0.75, (0, 0, 0), 2)
            self.overlay_image(self.video_frame_ref, self.image)
            self.decrement_persistance_counter()
        else:
            PERSIST_MESSAGE_FLAG = False
            PERSIST_MESSAGE_COUNTER = 0    

class ImageReader(object):
    def __init__(self, file_names):
        self.file_names = file_names
        self.max_idx = len(file_names)

    def __iter__(self):
        self.idx = 0
        return self

    def __next__(self):
        if self.idx == self.max_idx:
            raise StopIteration
        img = cv2.imread(self.file_names[self.idx], cv2.IMREAD_COLOR)
        if img.size == 0:
            raise IOError('Image {} cannot be read'.format(self.file_names[self.idx]))
        self.idx = self.idx + 1
        return img


class VideoReader(object):
    def __init__(self, file_name):
        self.file_name = file_name
        try:  # OpenCV needs int to read from webcam
            self.file_name = int(file_name)
        except ValueError:
            pass

    def __iter__(self):
        self.cap = cv2.VideoCapture(self.file_name)
        if not self.cap.isOpened():
            raise IOError('Video {} cannot be opened'.format(self.file_name))
        return self

    def __next__(self):
        was_read, img = self.cap.read()
        if not was_read:
            raise StopIteration
        return img


def infer_fast(net, img, net_input_height_size, stride, upsample_ratio, cpu,
               pad_value=(0, 0, 0), img_mean=(128, 128, 128), img_scale=1/256):
    height, width, _ = img.shape
    scale = net_input_height_size / height
    img_scale = 1/net_input_height_size

    scaled_img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    scaled_img = normalize(scaled_img, img_mean, img_scale)
    min_dims = [net_input_height_size, max(scaled_img.shape[1], net_input_height_size)]
    padded_img, pad = pad_width(scaled_img, stride, pad_value, min_dims)

    tensor_img = torch.from_numpy(padded_img).permute(2, 0, 1).unsqueeze(0).float()
    if not cpu:
        tensor_img = tensor_img.cuda()

    stages_output = net(tensor_img)

    stage2_heatmaps = stages_output[-2]
    heatmaps = np.transpose(stage2_heatmaps.squeeze().cpu().data.numpy(), (1, 2, 0))
    heatmaps = cv2.resize(heatmaps, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)

    stage2_pafs = stages_output[-1]
    pafs = np.transpose(stage2_pafs.squeeze().cpu().data.numpy(), (1, 2, 0))
    pafs = cv2.resize(pafs, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)

    return heatmaps, pafs, scale, pad


def overlay_image(img, img_overlay):
    img[0:img_overlay.shape[0], img.shape[1]-img_overlay.shape[1]:img.shape[1]] = img_overlay

PERSIST_MESSAGE_FLAG = False
PERSIST_MESSAGE_COUNTER = 0

state = State()
activity = Activity('virabhadrasana', './sample_poses/virabhadrasana/', 5)
activity.add_stage_instruction(1, 'Feet together, arms on hip')    
activity.add_stage_correction_check(stg_no=1, correction_text='bring your feet together', check_fn=check_if_feet_are_together)
activity.add_stage_correction_check(stg_no=1, correction_text='please rest arms on hip', check_fn=check_if_arms_on_hip)   

activity.add_stage_instruction(2, 'Spread feet, 2 feet apart in each direction')
activity.add_stage_correction_check(stg_no=2, correction_text='please rest arms on hip', check_fn=check_if_arms_on_hip)
activity.add_stage_correction_check(stg_no=2, correction_text='spread feet apart', check_fn=check_if_feet_are_apart)

activity.add_stage_instruction(3, 'Twist hips to the right, twist right foot to the right')
activity.add_stage_correction_check(stg_no=3, correction_text='Twist hip to the right', check_fn=check_hip_twist)

activity.add_stage_instruction(4, 'Raise arms')
activity.add_stage_correction_check(stg_no=4, correction_text='Raise your arms', check_fn=check_arm_raise)

activity.add_stage_instruction(5, 'Lower hips to a lunge position')
activity.add_stage_correction_check(stg_no=5, correction_text='lower into a lunge', check_fn=check_lunge_angle)
activity.add_stage_correction_check(stg_no=5, correction_text='Try to keep a straight leg', check_fn=check_back_leg_is_straight)
activity.add_stage_correction_check(stg_no=5, correction_text='keep arms raised', check_fn=check_arm_raise)

persist_message_instance = PersistMessage()
controller = Controller(state, activity, persist_message_instance)

def run_demo(net, image_provider, height_size, cpu, track_ids):
    net = net.eval()
    if not cpu:
        net = net.cuda()

    stride = 8
    upsample_ratio = 4
    num_keypoints = Pose.num_kpts
    previous_poses = []

    for img in image_provider:
        if(PERSIST_MESSAGE_FLAG==False):
            orig_img = img.copy()
            heatmaps, pafs, scale, pad = infer_fast(net, img, height_size, stride, upsample_ratio, cpu)

            total_keypoints_num = 0
            all_keypoints_by_type = []
            for kpt_idx in range(num_keypoints):  # 19th for bg
                total_keypoints_num += extract_keypoints(heatmaps[:, :, kpt_idx], all_keypoints_by_type, total_keypoints_num)

            pose_entries, all_keypoints = group_keypoints(all_keypoints_by_type, pafs, demo=True)
            
            for kpt_id in range(all_keypoints.shape[0]):
                all_keypoints[kpt_id, 0] = (all_keypoints[kpt_id, 0] * stride / upsample_ratio - pad[1]) / scale
                all_keypoints[kpt_id, 1] = (all_keypoints[kpt_id, 1] * stride / upsample_ratio - pad[0]) / scale
            
            for n in range(len(pose_entries)):
                if len(pose_entries[n]) == 0:
                    continue
                kpts_detected = np.zeros(num_keypoints, dtype=np.int32)    
                pose_keypoints = np.ones((num_keypoints, 2), dtype=np.int32) * -1
                for kpt_id in range(num_keypoints):
                    if pose_entries[n][kpt_id] != -1.0:  
                        kpts_detected[kpt_id] = 1
                        pose_keypoints[kpt_id, 0] = int(all_keypoints[int(pose_entries[n][kpt_id]), 0])
                        pose_keypoints[kpt_id, 1] = int(all_keypoints[int(pose_entries[n][kpt_id]), 1])
                
                controller.update_video_frame_reference(img)
                controller.check_if_frames_are_consistent(pose_keypoints, kpts_detected)
                if(controller.State.current_stage>=1):
                    pose = Pose(pose_keypoints, pose_entries[n][18])
                    #pose.draw(img)

            img = cv2.addWeighted(orig_img, 0.3, img, 1, 0)

            controller.update_video_frame_reference(img)                    
        else:
            controller.update_video_frame_reference(img)
            controller.PersistMessage.persist()

        cv2.imshow('User Activity Guidance', img)

        key = cv2.waitKey(33)
        if key == 27:  # esc
            return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='''Lightweight human pose estimation python demo.
                       This is just for quick results preview.
                       Please, consider c++ demo for the best performance.''')
    parser.add_argument('--checkpoint-path', type=str, default='./checkpoint_iter_370000.pth', help='path to the checkpoint')
    parser.add_argument('--height-size', type=int, default=128, help='network input layer height size')
    parser.add_argument('--video', type=str, default='0', help='path to video file or camera id')
    parser.add_argument('--cpu', default=True, action='store_true', help='run network inference on cpu')
    parser.add_argument('--track-ids', default=False, help='track poses ids')
    args = parser.parse_args()

    net = PoseEstimationWithMobileNet()
    checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
    load_state(net, checkpoint)

    frame_provider = VideoReader(args.video)

    run_demo(net, frame_provider, args.height_size, args.cpu, args.track_ids)
