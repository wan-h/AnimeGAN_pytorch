# coding: utf-8
# Author: wanhui0729@gmail.com

import os

def frames2video(frame_path, out_path):
    video_output_path = os.path.join(out_path, os.path.basename(frame_path))
    os.system("ffmpeg -f image2 -i {}/%d.jpg  -vcodec libx264  -r 24 {} -y".format(frame_path, video_output_path))

if __name__ == '__main__':
    frames_path = "/data/SRC/hayao/video1/dst"
    out_path = "/data/SRC/hayao/video1/videos"
    frames_path_list = os.listdir(frames_path)
    for fp in frames_path_list:
        frame_path = os.path.join(frames_path, fp)
        frames2video(frame_path, out_path)