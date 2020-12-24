# coding: utf-8
# Author: wanhui0729@gmail.com

import os
import cv2
from tqdm import tqdm

# 使用1080P的画质
TARGET_SIZE = (1920, 1080)

def save_frames_from_video(video, output, interval):
    if not os.path.exists(output):
        os.makedirs(output)
    videoCapture = cv2.VideoCapture(video)
    fps = int(videoCapture.get(cv2.CAP_PROP_FPS))
    frame_num = int(videoCapture.get(cv2.CAP_PROP_FRAME_COUNT))
    interval_frames = fps * interval
    count = 0
    for _ in tqdm(range(frame_num)):
        success, frame = videoCapture.read()
        if success:
            count += 1
        if count % interval_frames == 0:
            frame_name = "{}.jpg".format(count)
            frame = cv2.resize(frame, TARGET_SIZE)
            cv2.imwrite(os.path.join(output, frame_name), frame)

if __name__ == '__main__':
    # 视频转图像
    save_frames_from_video(
        video='/data/datasets/animegan/your_name.mkv',
        output='/data/datasets/animegan/you_name_more',
        interval=1
    )