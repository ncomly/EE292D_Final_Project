from video2tfrecord import convert_videos_to_tfrecord

in_path = "/mnt/disks/data/dataset/lipread_mp4/ACTUALLY/train/"
#in_path = "/mnt/disks/data/dataset/lipread_mp4/DETAILS/train/DETAILS_01000.mp4"
out_path = "./test_tfrecord_ACTUALLY_color"
#out_path = "."
#convert_videos_to_tfrecord(source_path = in_path, destination_path = out_path, width=256, height=256, n_videos_in_record=1, n_frames_per_video=29, file_suffix="*.mp4", dense_optical_flow=False)
#convert_videos_to_tfrecord(source_path = in_path, destination_path = out_path, width=256, height=256, n_videos_in_record=1, n_frames_per_video=29, file_suffix="*.mp4", dense_optical_flow=False)
convert_videos_to_tfrecord(source_path = in_path, destination_path = out_path, width=96, height=96, n_videos_in_record=1, n_frames_per_video=29, file_suffix="*.mp4", dense_optical_flow=False)
