# coding: utf-8
# Author: wanhui0729@gmail.com

import os
import cv2
import uuid
from tqdm import tqdm

TARGET_SIZE = (960, 540)

def save_frames_from_video(video, output, interval, randomName=False):
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
            if randomName:
                frame_name = "{}.jpg".format(str(uuid.uuid4()))
            else:
                frame_name = "{}.jpg".format(count)
            frame = cv2.resize(frame, TARGET_SIZE)
            cv2.imwrite(os.path.join(output, frame_name), frame)

def resize_images(images_path):
    for image in tqdm(os.listdir(images_path)):
        image_path = os.path.join(images_path, image)
        img = cv2.imread(image_path)
        img = cv2.resize(img, (480, 270))
        cv2.imwrite(image_path, img)

def save_frames_from_videos(videos, output, interval):
    for video_name in os.listdir(videos):
        video_path = os.path.join(videos, video_name)
        output_path = os.path.join(output, video_name)
        os.mkdir(output_path)
        save_frames_from_video(video_path, output_path, interval, randomName=True)

def mv_data(videos_frames, output):
    for video_frames in os.listdir(videos_frames):
        video_path = os.path.join(videos_frames, video_frames)
        os.system("cp {}/* {}".format(video_path, output))

if __name__ == '__main__':
    # 视频转图像
    # save_frames_from_video(
    #     video='/data/datasets/animegan/your_name.mkv',
    #     output='/data/datasets/animegan/you_name',
    #     interval=1
    # )

    # resize_images('/data/datasets/animegan/ip11/train/t')

    # save_frames_from_videos(
    #     videos="/data/SRC/shinkai/video_1/src",
    #     output="/data/datasets/animegan/ip11/train/tmp",
    #     interval=1
    # )

    mv_data(
        videos_frames="/data/datasets/animegan/ip11/test/tmp",
        output="/data/datasets/animegan/ip11/test/real"
    )
    pass