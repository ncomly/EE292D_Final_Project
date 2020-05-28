from video2tfrecord import convert_videos_to_tfrecord

import glob, os

#in_path = "/mnt/disks/data/dataset/lipread_mp4/ACTUALLY/train/"
#out_path = "./test_tfrecord_ACTUALLY_color"
#convert_videos_to_tfrecord(source_path = in_path, destination_path = out_path, width=96, height=96, n_videos_in_record=1, n_frames_per_video=29, file_suffix="*.mp4", dense_optical_flow=False)

folds = ["train", "val", "test"]
with open('label_subset.txt','r') as fp:
    word = fp.readline().strip()
    while (word):
        print("Processing: ",word)
        word_dir = "/mnt/disks/data/dataset/lipread_tfrecords/"+word
        if not os.path.isdir(word_dir):
            os.mkdir(word_dir)
        for fold in folds:
            in_path = "/mnt/disks/data/dataset/lipread_mp4/"+word+"/"+fold+"/"
            out_path = "/mnt/disks/data/dataset/lipread_tfrecords/"+word+"/"+fold
            if not os.path.isdir(out_path): 
                os.mkdir(out_path)
            print(in_path)
            print(out_path)
            convert_videos_to_tfrecord(source_path = in_path, destination_path = out_path, width=96, height=96, n_videos_in_record=1, n_frames_per_video=29, file_suffix="*.mp4", dense_optical_flow=False)
        word = fp.readline().strip()

print("Conversion to tfrecords complete")
