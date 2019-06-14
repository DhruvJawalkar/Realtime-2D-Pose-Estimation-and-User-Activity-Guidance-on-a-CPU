class State:
    def __init__(self):
        self.current_stage = 0
        self.current_instruction = 0
        self.activity_completed = False
    
    def set_current_stage(self, stage):
        self.current_stage = stage
        self.current_instruction = 1

class Activity:
    def __init__(self, name='virabhadrasana', im_dir='./sample_poses/virabhadrasana/', n_stages = 5):
        self.name = name,
        self.pose_im_dir = im_dir,
        self.n_stages = n_stages 
        self.rules = {'stage'+str(i+1) : {} for i in range(n_stages)}

    def add_stage_instruction(self, stg_no, instruction_text):
        stg = self.rules['stage'+str(stg_no)]
        stg['instruction_text'] = ''#instruction_text
        stg['correction_checks'] = []

    def add_stage_correction_check(self, stg_no, correction_text, check_fn):
        correction = {}
        correction = {
            'correction_text' : correction_text,
            'check_fn' : check_fn
        }

        self.rules['stage'+str(stg_no)]['correction_checks'].append(correction)
