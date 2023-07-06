from threading import Thread
from copy import deepcopy
import time

import numpy as np
import cv2


class Streamer:
    def __init__(self, cam_id: int, vc_size=(640, 480), rotation_degree=0):
        # def vc
        self.vc_size = np.array(vc_size, dtype=np.uint16)
        self.vc = cv2.VideoCapture(cam_id)
        self.vc.set(cv2.CAP_PROP_FRAME_WIDTH, self.vc_size[0])
        self.vc.set(cv2.CAP_PROP_FRAME_HEIGHT, self.vc_size[1])
        # init status
        self.cam_state = False
        self.frame = None
        # set rotation
        self.rotation_degree = rotation_degree
        self.out_size = np.array([max(self.vc_size)] * 2, dtype=np.uint16)
        self.padding_w, self.padding_h = (self.out_size - self.vc_size) // 2
        self.mat_rotate = cv2.getRotationMatrix2D(center=self.out_size//2,
                                                  angle=self.rotation_degree, scale=1)  # 2x3
        # creat threading
        self.thread = Thread(name='camera', target=self.update, daemon=True)  # open threading till main killed
        self.thread.start()
        print('camera threading start')
        # wait for camera to open
        time.sleep(2)

    def update(self):
        if self.vc.isOpened():
            self.cam_state = True
        while self.cam_state:
            self.cam_state, self.frame = self.vc.read()

    def grab_frame(self):
        """
        return:
            self.rotation_degree == 0:
                original frame
            else:
                padding rectangle frame with its maximum side length
        """
        if self.cam_state:
            if self.rotation_degree:
                rectangle_frame = cv2.copyMakeBorder(self.frame, self.padding_h, self.padding_h,
                                                     self.padding_w, self.padding_w, cv2.BORDER_CONSTANT)
                res_frame = cv2.warpAffine(rectangle_frame, self.mat_rotate, self.out_size, borderValue=0)
            else:
                res_frame = deepcopy(self.frame)
            return res_frame
        else:
            return None


if __name__ == '__main__':
    streamer = Streamer(1, vc_size=(320, 240), rotation_degree=0)
    while True:
        frame = streamer.grab_frame()
        if frame is not None:
            cv2.imshow('frame', frame)
            k = cv2.waitKey(1) & 0xFF
            if k == 27:  # ESC out
                cv2.destroyAllWindows()
                break
        else:
            break
    print('over')
