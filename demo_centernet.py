#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
import copy
import argparse

import cv2 as cv
import numpy as np
import onnxruntime


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--file", type=str, default=None)
    parser.add_argument("--width", help='cap width', type=int, default=960)
    parser.add_argument("--height", help='cap height', type=int, default=540)

    parser.add_argument("--model",
                        default='model/centernet_resnet50_v1_fpn_512x512.onnx')
    parser.add_argument(
        "--input_size",
        type=str,
        default='512,512',
    )
    parser.add_argument("--score_th", type=float, default=0.6)

    args = parser.parse_args()

    return args


def main():
    # 引数解析 #################################################################
    args = get_args()
    cap_device = args.device
    cap_width = args.width
    cap_height = args.height

    model_path = args.model
    input_shape = [int(i) for i in args.input_size.split(',')]
    score_th = args.score_th

    if args.file is not None:
        cap_device = args.file

    # カメラ準備 ###############################################################
    cap = cv.VideoCapture(cap_device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    # モデルロード #############################################################
    onnx_session = onnxruntime.InferenceSession(
        model_path,
        providers=['CPUExecutionProvider'],
    )

    while True:
        start_time = time.time()

        # カメラキャプチャ #####################################################
        ret, frame = cap.read()
        if not ret:
            continue
        frame_width, frame_height = frame.shape[1], frame.shape[0]
        debug_image = copy.deepcopy(frame)

        # 検出実施 #############################################################
        input_image = cv.resize(frame, dsize=(input_shape[1], input_shape[0]))
        input_image = cv.cvtColor(input_image, cv.COLOR_BGR2RGB)
        input_image = np.expand_dims(input_image, axis=0).astype('uint8')
        # input_image = np.expand_dims(input_image, axis=0).astype('float32')  # centernet_mobilenetv2_fpn_od

        input_name = onnx_session.get_inputs()[0].name
        result = onnx_session.run(None, {input_name: input_image})

        num_detections = int(result[3][0])
        detection_classes = result[1][0]
        detection_boxes = result[0][0]
        detection_scores = result[2][0]
        # num_detections = int(result[0][0])  # centernet_mobilenetv2_fpn_od
        # detection_classes = result[2][0]  # centernet_mobilenetv2_fpn_od
        # detection_boxes = result[3][0]  # centernet_mobilenetv2_fpn_od
        # detection_scores = result[1][0]  # centernet_mobilenetv2_fpn_od

        elapsed_time = time.time() - start_time

        for index in range(num_detections):
            score = detection_scores[index]
            bbox = detection_boxes[index]
            class_id = int(detection_classes[index])

            if score < score_th:
                continue

            # 検出結果可視化 ###################################################
            x1, y1 = int(bbox[1] * frame_width), int(bbox[0] * frame_height)
            x2, y2 = int(bbox[3] * frame_width), int(bbox[2] * frame_height)

            cv.putText(debug_image,
                       'ID:' + str(class_id) + '({:.3f})'.format(score),
                       (x1, y1 - 15), cv.FONT_HERSHEY_SIMPLEX, 0.6,
                       (0, 255, 0), 2, cv.LINE_AA)
            cv.rectangle(debug_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv.putText(
            debug_image,
            "Elapsed Time:" + '{:.1f}'.format(elapsed_time * 1000) + "ms",
            (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv.LINE_AA)

        # キー処理(ESC：終了) #################################################
        key = cv.waitKey(1)
        if key == 27:  # ESC
            break

        # 画面反映 #############################################################
        cv.imshow('Object Detection API ONNX Demo', debug_image)

    cap.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
