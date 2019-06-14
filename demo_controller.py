import cv2 
import numpy as np

class Controller():
    def __init__(self, State, Activity, PersistMessage):
        self.State = State
        self.Activity = Activity
        self.PersistMessage = PersistMessage
        self.poses_overview = cv2.imread('./sample_poses/virabhadrasana/collage-resized.png')
        self.stg1_img = cv2.imread('./sample_poses/virabhadrasana/stage1-resized.png')
        self.stg2_img = cv2.imread('./sample_poses/virabhadrasana/stage2-resized.png')
        self.stg3_img = cv2.imread('./sample_poses/virabhadrasana/stage3-resized.png')
        self.stg4_img = cv2.imread('./sample_poses/virabhadrasana/stage4-resized.png')
        self.stg5_img = cv2.imread('./sample_poses/virabhadrasana/stage5-resized.png')
        self.kps_diff_threshold = 200
        self.ref_frame_kps = np.ones((18, 2), dtype=np.int32) * -1
        self.video_frame_ref = ''
        self.kpt_names = ['nose', 'neck',
        'r_sho', 'r_elb', 'r_wri', 'l_sho', 'l_elb', 'l_wri',
        'r_hip', 'r_knee', 'r_ank', 'l_hip', 'l_knee', 'l_ank',
        'r_eye', 'l_eye', 'r_ear', 'l_ear']
        self.minimal_kps = np.array([0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 
        1, 0, 0, 0, 0], dtype=np.int32)
        self.minimal_kps_mask = self.minimal_kps.astype(np.bool)

    def update_video_frame_reference(self, video_frame_ref):
        self.video_frame_ref = video_frame_ref
        self.PersistMessage.update_video_frame_ref(self.video_frame_ref)

    def display_text_in_video_ref_frame(self, text, pos=(20,20)):
        cv2.putText(self.video_frame_ref, text, pos, cv2.FONT_HERSHEY_COMPLEX, 0.75, (0, 0, 0), 2)
    
    def display_correction_text(self, correction_text, pos=(20,70)):
        cv2.putText(self.video_frame_ref, correction_text, pos, cv2.FONT_HERSHEY_COMPLEX, 0.75, (0, 0, 0), 2)

    def set_ref_frame_kps(self, kps):
        self.ref_frame_kps = kps    
    
    def get_difference_between_kps(self, ref_frame_kps, cur_frame_kps, mask=None):
        return np.abs(np.sum(ref_frame_kps[mask] - cur_frame_kps[mask]))

    def are_minimal_kps_visible(self, cur_visible_kps, minimal_kps):
        return np.all(np.logical_and(cur_visible_kps, minimal_kps) == minimal_kps)

    def overlay_image(self, img_overlay):
        video_frame = self.video_frame_ref
        video_frame[0:img_overlay.shape[0], video_frame.shape[1]-img_overlay.shape[1]:video_frame.shape[1]] = img_overlay

    def check_if_frames_are_consistent(self, cur_frame_kps, cur_visible_kps):
        cur_stg = self.State.current_stage

        if(cur_stg==0 and not self.are_minimal_kps_visible(cur_visible_kps, self.minimal_kps)):
            self.display_text_in_video_ref_frame('Please step back, make sure you are completely visible in the video frame.')
            return False

        elif(cur_stg==0):
            if(self.get_difference_between_kps(self.ref_frame_kps, cur_frame_kps, self.minimal_kps_mask) <= self.kps_diff_threshold):
                # begin guidance
                self.State.set_current_stage(1)
                self.PersistMessage.set_message(self.video_frame_ref, 'Virabhadrasana', self.poses_overview, 100) 
                return True
            else:
                self.set_ref_frame_kps(cur_frame_kps)
                return False 
        
        elif(cur_stg==1):
            self.overlay_image(self.stg1_img)
            self.display_text_in_video_ref_frame(self.Activity.rules['stage'+str(cur_stg)]['instruction_text'])
            corrections = self.Activity.rules['stage'+str(cur_stg)]['correction_checks']

            if(not corrections[0]['check_fn'](cur_frame_kps)):
                self.display_correction_text(corrections[0]['correction_text'])
                return    
            if(not corrections[1]['check_fn'](cur_frame_kps)):
                self.display_correction_text(corrections[1]['correction_text'])
                return
            self.State.set_current_stage(2)    

        elif(cur_stg==2):
            self.overlay_image(self.stg2_img)
            self.display_text_in_video_ref_frame(self.Activity.rules['stage'+str(cur_stg)]['instruction_text'])
            corrections = self.Activity.rules['stage'+str(cur_stg)]['correction_checks']

            if(not corrections[0]['check_fn'](cur_frame_kps)):
                self.display_correction_text(corrections[0]['correction_text'])
                return    
            if(not corrections[1]['check_fn'](cur_frame_kps)):
                self.display_correction_text(corrections[1]['correction_text'])
                return
            self.State.set_current_stage(3)           

        elif(cur_stg==3):
            self.overlay_image(self.stg3_img)
            self.display_text_in_video_ref_frame(self.Activity.rules['stage'+str(cur_stg)]['instruction_text'])
            corrections = self.Activity.rules['stage'+str(cur_stg)]['correction_checks']

            if(not corrections[0]['check_fn'](cur_frame_kps)):
                self.display_correction_text(corrections[0]['correction_text'])
                return    
            self.State.set_current_stage(4)  
        
        elif(cur_stg==4):
            self.overlay_image(self.stg4_img)
            self.display_text_in_video_ref_frame(self.Activity.rules['stage'+str(cur_stg)]['instruction_text'])
            corrections = self.Activity.rules['stage'+str(cur_stg)]['correction_checks']

            if(not corrections[0]['check_fn'](cur_frame_kps)):
                self.display_correction_text(corrections[0]['correction_text'])
                return    
            self.State.set_current_stage(5)
        
        elif(cur_stg==5):
            self.overlay_image(self.stg5_img)
            self.display_text_in_video_ref_frame(self.Activity.rules['stage'+str(cur_stg)]['instruction_text'])
            corrections = self.Activity.rules['stage'+str(cur_stg)]['correction_checks']

            if(not corrections[0]['check_fn'](cur_frame_kps)):
                self.display_correction_text(corrections[0]['correction_text'])
                return
            if(not corrections[1]['check_fn'](cur_frame_kps)):
                self.display_correction_text(corrections[1]['correction_text'])
                return
            if(not corrections[2]['check_fn'](cur_frame_kps)):
                self.display_correction_text(corrections[2]['correction_text'])
                return    
            self.display_correction_text('Activity done.')
            self.State.current_stage = -1    
            self.State.activity_completed = True
        elif(self.State.activity_completed):
            self.display_correction_text('Activity done.')

