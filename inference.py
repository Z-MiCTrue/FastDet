import time

import torch
import numpy as np
import cv2
import onnx
from onnxsim import simplify

from module.FastestDet import FastestDet
from utils.camera import Streamer
from utils.general_transform import img_resize
from configs import Parameters


def use_NMS(data_ori, IoU_max):
    # 按列分割 x_1, y_1, x_2, y_2, scores
    x_1, y_1, x_2, y_2 = np.split(data_ori[:, :4], 4, axis=1)
    scores = data_ori[:, 4]
    # score_index_list是按照score最大值降序排序的索引列表
    score_index_list = np.argsort(-scores, axis=0).flatten()
    # 每一个候选框的面积
    each_areas = (x_2 - x_1) * (y_2 - y_1)
    bbox_keep = []
    while score_index_list.size > 0:
        # 当前置信度最高加入保留组
        score_index = score_index_list[0]
        bbox_keep.append(score_index)
        # 计算当前概率矩形框与其他矩形框的相交框的坐标->得到的是数组(1-n维逐个比较)
        x_overlay_1 = np.maximum(x_1[score_index], x_1[score_index_list[1:]])
        y_overlay_1 = np.maximum(y_1[score_index], y_1[score_index_list[1:]])
        x_overlay_2 = np.minimum(x_2[score_index], x_2[score_index_list[1:]])
        y_overlay_2 = np.minimum(y_2[score_index], y_2[score_index_list[1:]])
        # 计算相交框的面积, 边长为负时用0代替
        overlay_w = np.maximum(0, x_overlay_2 - x_overlay_1)
        overlay_h = np.maximum(0, y_overlay_2 - y_overlay_1)
        area_overlay = overlay_w * overlay_h
        # 计算重叠度IOU：重叠面积 / (面积1 + 面积2 - 重叠面积)
        IoU = area_overlay / (each_areas[score_index] + each_areas[score_index_list[1:]] - area_overlay)
        # 找到重叠度低于阈值的矩形框索引并生成下一索引(除去当前+1)
        next_index = np.where(IoU <= IoU_max)[0] + 1
        # 将score_index_list序列更新, 仅保留过限bbox
        score_index_list = score_index_list[next_index]
    return data_ori[bbox_keep]


# 后处理 (归一化后的坐标)
def post_process(preds: torch.Tensor, conf_thresh=0.1, nms_thresh=0.3):
    output_bboxes = []
    for bboxes_np in preds.numpy():
        bboxes_np = bboxes_np[bboxes_np[:, 4] >= conf_thresh]
        res_bboxes = torch.from_numpy(use_NMS(bboxes_np, nms_thresh))
        output_bboxes.append(res_bboxes)
    return output_bboxes


def frame_process(model, frame, cfg):
    frame = img_resize(frame, cfg.input_size, keep_ratio=True)
    frame_in = torch.from_numpy(np.transpose(np.expand_dims(frame, axis=0), (0, 3, 1, 2))).to(cfg.device).float() / 255
    # 推理
    with torch.no_grad():
        preds = model(frame_in).detach().to('cpu')
    # 后处理
    res_bbox = post_process(preds, conf_thresh=cfg.conf_thresh, nms_thresh=cfg.nms_thresh)[0].numpy()
    res_bbox[:, : 4] *= [cfg.input_size[0], cfg.input_size[1], cfg.input_size[0], cfg.input_size[1]]
    print(res_bbox)
    # 显示
    for each_bbox in res_bbox:
        # 画框
        cv2.rectangle(frame, tuple(each_bbox[:2].astype(np.int16)), tuple(each_bbox[2: 4].astype(np.int16)),
                      (255, 0, 0))
        # 获得最大概率类别索引
        class_index = int(each_bbox[5])
        # 获得最大概率类别概率值
        class_possible = str(np.round(each_bbox[4], 4))
        cv2.putText(frame, f'{cfg.classes_name[class_index]} {class_possible}',
                    tuple(each_bbox[:2].astype(np.int16)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0))
    return frame


def check_model(cfg):
    model = FastestDet(cfg.category_num, True).to(cfg.device)
    model.load_state_dict(torch.load(cfg.test_weight_path, map_location=cfg.device))
    model.eval()

    if cfg.test_img:
        frame = cv2.imread(cfg.img_path, 1)
        frame = frame_process(model, frame, cfg)
        cv2.imshow('frame', frame)
        cv2.waitKey()
    else:
        streamer = Streamer(cfg.camera_id)
        frame = streamer.grab_frame()
        fps = 0
        timestamp = time.time()
        while frame is not None:
            # 计算fps
            fps += 1
            now = time.time()
            if now - timestamp >= 1:
                timestamp = now
                print('fps:', '=' * fps, fps)
                fps = 0
            frame = frame_process(model, frame, cfg)
            cv2.imshow('frame', frame)
            k = cv2.waitKey(1) & 0xFF
            if k == 27:  # ESC out
                cv2.destroyAllWindows()
                break
            else:
                frame = streamer.grab_frame()


# 导出onnx模型
def export_onnx(cfg):
    device = torch.device('cpu')
    export_path = './weights/FastestDet.onnx'

    model = FastestDet(cfg.category_num, True).to(device)
    model.load_state_dict(torch.load(cfg.test_weight_path, map_location=device))
    model.eval()

    input_sample = torch.ones((1, 3, cfg.input_height, cfg.input_width),
                              dtype=torch.float32, requires_grad=True, device=device)
    torch.onnx.export(model, input_sample, export_path,
                      input_names=['input'], output_names=['output'],
                      opset_version=11,  # the ONNX version to export the model to
                      do_constant_folding=True)  # whether to execute constant folding for optimization
    print('finished exporting onnx')
    # onnx简化
    print('start simplifying onnx')
    onnx_model = onnx.load(export_path)  # load onnx model
    model_simp, flag = simplify(onnx_model)
    if flag:
        onnx.save(model_simp, export_path)
        print("simplify onnx successfully")
    else:
        print("simplify onnx failed")


if __name__ == '__main__':
    paras = Parameters()
    check_model(paras)
    # export_onnx(paras)
