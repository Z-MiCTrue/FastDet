import numpy as np
import cv2


#  缩放
def img_resize(img: np.ndarray, size: tuple, keep_ratio=True, points: list = None):
    """
    size=(w, h) tuple
    points:[[x_1, y_1, ...],] list
    """
    h_ori, w_ori, channel = img.shape[:3]
    w_new, h_new = size
    # 需要补边
    if keep_ratio and w_new / w_ori != h_new / h_ori:
        scale = min(w_new / w_ori, h_new / h_ori)
        w_valid, h_valid = round(w_ori * scale), round(h_ori * scale)
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        # 补边填充
        aim_size = np.zeros((h_new, w_new, channel), dtype=np.uint8)
        h_padding = np.abs(h_new - h_valid) // 2
        w_padding = np.abs(w_new - w_valid) // 2
        aim_size[h_padding: h_valid + h_padding, w_padding: w_valid + w_padding] = img
        # 点转换
        if points is None:
            return aim_size
        else:
            points = np.array(points)
            points = points * np.tile(np.array([scale, scale]), points.shape[-1] // 2) + \
                     np.tile(np.array([w_padding, h_padding]), points.shape[-1] // 2)
            return aim_size, points.tolist()
    # 不需要改变
    elif w_new == w_ori and h_new == h_ori:
        if points is None:
            return img
        else:
            return img, points
    # 不需成比例或已成比例
    else:
        aim_size = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
        if points is None:
            return aim_size
        else:
            fx = w_new / w_ori
            fy = h_new / h_ori
            points = np.array(points)
            points = points * np.tile(np.array([fx, fy]), points.shape[-1] // 2)
            return aim_size, points.tolist()


# 单目相机矫正
class Mono_Correct(object):
    def __init__(self, innerMat: np.ndarray, dist: np.ndarray):  # 内参矩阵, 畸变系数
        self.internal_matrix = innerMat  # matlab计算的内参矩阵需要转置
        self.distortion_coefficient = dist

    def correction_calculation(self, img: np.ndarray):
        if self.internal_matrix is not None and self.distortion_coefficient is not None:
            img_h, img_w = img.shape[:2]
            newMat, roi = cv2.getOptimalNewCameraMatrix(self.internal_matrix, self.distortion_coefficient,
                                                        (img_w, img_h), 1, (img_w, img_h))
            img = cv2.undistort(img, self.internal_matrix, self.distortion_coefficient, None, newMat)
        return img


# 透视变换
class Perspective_Transform(Mono_Correct):
    """
    in_loc/out_loc: [[x, y]] np.ndarray shape=(4, 2) dtype=np.float32
    """
    def __init__(self, in_loc: np.ndarray, out_loc: np.ndarray, innerMat: np.ndarray = None, dist: np.ndarray = None):
        # 继承畸变矫正变换
        super(Perspective_Transform, self).__init__(innerMat, dist)
        # 逆透视变换 (cv2.warpAffine仿射变换使用2x3变换矩阵; cv2.warpPerspective透视变换使用3x3变换矩阵缩放)
        # 生成逆透视变换矩阵(后四点(x, y), 前四点(x, y), float32)
        self.transform_matrix = cv2.getPerspectiveTransform(in_loc, out_loc)
        self.scale = np.max(out_loc, axis=0).astype(np.int16)

    def refresh_loc(self, in_loc: np.ndarray, out_loc: np.ndarray):
        self.transform_matrix = cv2.getPerspectiveTransform(in_loc, out_loc)
        self.scale = np.max(out_loc, axis=0).astype(np.int16)

    def perspective_calculate(self, img: np.ndarray):
        img_calibrated = cv2.warpPerspective(img, self.transform_matrix, dsize=self.scale)
        return img_calibrated


if __name__ == '__main__':
    i_loc = np.float32([[0, 0], [320, 0],
                        [0, 320], [320, 320]])
    o_loc = np.float32([[0, 0], [100, 0],
                        [0, 100], [100, 100]])
    perspective_transform = Perspective_Transform(i_loc, o_loc)

    frame = cv2.imread('../data/test.jpg', 1)
    frame_transformed = perspective_transform.perspective_calculate(frame)
    cv2.imshow('', frame_transformed)
    cv2.waitKey()
