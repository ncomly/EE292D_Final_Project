from video2tfrecord import convert_videos_to_tfrecord

in_path = "/mnt/disks/data/dataset/lipread_mp4/DETAILS/train/"
out_path = "./test_tf_record_gray"
#convert_videos_to_tfrecord(source_path = in_path, destination_path = out_path, width=256, height=256, n_videos_in_record=1, n_frames_per_video=29, file_suffix="*.mp4", dense_optical_flow=False)
convert_videos_to_tfrecord(source_path = in_path, destination_path = out_path, width=256, height=256, n_videos_in_record=1, n_frames_per_video=29, file_suffix="*.mp4", dense_optical_flow=False)
