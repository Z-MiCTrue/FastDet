import torch


class Parameters:
    def __init__(self):
        # DATASET
        self.train_txt = './data/Darknet_Yolo/train.txt'
        self.val_txt = './data/Darknet_Yolo/val.txt'
        self.names = './data/Darknet_Yolo/category.names'
        with open(self.names, 'r') as f:
            self.classes_name = f.read().strip().split('\n')
            print(f'classes_name: {self.classes_name}')
        # MODEL
        self.category_num = 1
        self.input_size = (320, 320)  # w, h
        self.input_width, self.input_height = self.input_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # TRAIN
        self.num_workers = 2
        self.learn_rate = 0.001
        self.weight_decay = 0.0005
        self.batch_size = 96
        self.end_epoch = 300
        self.milestones = [150, 200, 250]
        self.pretrained_weight = None
        # POSTPROCESS
        self.conf_thresh = 0.8
        self.nms_thresh = 0.3
        # INFERENCE TEST&EXPORT
        self.test_weight_path = './weights/weight_best_AP05-0.8_epoch-130.pth'
        self.test_img = True
        self.img_path = './data/test2.jpg'
        self.camera_id = 1
