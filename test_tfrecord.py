from video2tfrecord import convert_videos_to_tfrecord

import glob

in_path = "/mnt/disks/data/dataset/lipread_mp4/ACTUALLY/train/"
out_path = "./test_tfrecord_ACTUALLY_color"
convert_videos_to_tfrecord(source_path = in_path, destination_path = out_path, width=96, height=96, n_videos_in_record=1, n_frames_per_video=29, file_suffix="*.mp4", dense_optical_flow=False)

