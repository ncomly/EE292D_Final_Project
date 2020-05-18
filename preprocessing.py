import numpy as np
import cv2
import os
import tqdm
from video2tfrecord import convert_videos_to_tfrecord
from tensorflow.python.platform import gfile

mp4_path = 'lipread_mp4/'
npy_path = 'lipread_npy/'


def ConvertToNPY(data_dir):
    # get all words
    mp4_data = data_dir + mp4_path
    for word in tqdm.tqdm(os.listdir(mp4_data)):
        word_path = data_dir+npy_path + word + '/'
        if not(os.path.isdir(word_path)):
            os.mkdir(word_path)
            for dset in ['train/', 'test/', 'val/']:
                dset_folder = data_dir + mp4_path + word + '/' + dset
                os.mkdir(data_dir+npy_path + word + '/' + dset)
                for file in os.listdir(dset_folder):
                    if file.endswith('.mp4'):
                        Save(Convert(dset_folder, file), dset_folder, file)


def Save(data, path, file):
    # save path
    save_path = path.replace(mp4_path, npy_path) 
    np.save(save_path + file.replace(".mp4", ".npy"), data)


def Convert(path, file):
    cap = cv2.VideoCapture(path + file)

    buf = np.empty((29, 256, 256), np.dtype('uint8')) # grayscale
    # buf = np.empty((29, 256, 256, 3), np.dtype('uint8'))

    fc = 0
    ret = True

    while (fc < 29  and ret):
        ret, frame = cap.read()

        # Capture frame-by-frame - grayscale
        buf[fc] = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # buf[fc] = frame

        fc += 1

    cap.release()

    # centercrop
    return CenterCrop(buf, (96,96))
    

# assumes in format (f, h, w, c)
def CenterCrop(vid, size):
    h,w = vid.shape[2], vid.shape[1]
    ch, cw = h//2, w//2
    nh, nw = size
    if vid.ndim == 4:
        return vid[:, ch-nh//2:ch+nh//2, cw-nw//2:cw+nw//2, :]
    else:
        return vid[:, ch-nh//2:ch+nh//2, cw-nw//2:cw+nw//2]


def ConvertTesting(path):
    cap = cv2.VideoCapture(path+'.mp4')
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(frameCount, frameWidth, frameHeight)

    buf = np.empty((frameCount, frameHeight, frameWidth, 3), np.dtype('uint8'))

    fc = 0
    ret = True

    while (fc < frameCount  and ret):
        # ret, buf[fc] = cap.read()

        # Capture frame-by-frame
        ret, buf[fc] = cap.read()

        fc += 1

    cap.release()

    cv2.namedWindow('frame 10')
    cv2.imshow('frame 10', buf[9])

    cv2.waitKey(0)

    # buf = buf.transpose(0,3,1,2)
    print(buf.shape)
    crop = CenterCrop(buf, (96,96))
    print(crop.shape)
    # buf = buf.transpose(0,2,3,1)

    cap.release()

    cv2.namedWindow('frame 10 - cropped')
    cv2.imshow('frame 10 - cropped', crop[9])

    cv2.waitKey(0)

    print(crop[0][0][0])
    gray = Grayscale(crop)
    print(gray.shape)
    print(gray[0][0][0])

    cv2.namedWindow('frame 10 - gray')
    cv2.imshow('frame 10 - gray', gray[9])
    cv2.waitKey(0)

    np.save(path+'.npy', gray)


# assumes in format (f, w, h, c)
def Grayscale(vid):
    f, h, w, c = vid.shape
    gray = np.empty((f, h, w), np.dtype('uint8'))
    fc = 0
    while fc < f:
        gray[fc] = cv2.cvtColor(vid[fc], cv2.COLOR_BGR2GRAY)
        fc += 1

    return gray

# def CenterCrop(batch_img, size):
#     w, h = batch_img[0][0].shape[1], batch_img[0][0].shape[0]
#     th, tw = size
#     img = np.zeros((len(batch_img), len(batch_img[0]), th, tw))
#     for i in range(len(batch_img)):
#         x1 = int(round((w - tw))/2.)
#         y1 = int(round((h - th))/2.)
#         img[i] = batch_img[i, :, y1:y1+th, x1:x1+tw]
#     return img


def RandomCrop(batch_img, size):
    w, h = batch_img[0][0].shape[1], batch_img[0][0].shape[0]
    th, tw = size
    img = np.zeros((len(batch_img), len(batch_img[0]), th, tw))
    for i in range(len(batch_img)):
        x1 = random.randint(0, 8)
        y1 = random.randint(0, 8)
        img[i] = batch_img[i, :, y1:y1+th, x1:x1+tw]
    return img


def HorizontalFlip(batch_img):
    for i in range(len(batch_img)):
        if random.random() > 0.5:
            for j in range(len(batch_img[i])):
                batch_img[i][j] = cv2.flip(batch_img[i][j], 1)
    return batch_img


def ColorNormalize(batch_img):
    mean = 0.413621
    std = 0.1700239
    batch_img = (batch_img - mean) / std
    return batch_img


# ConvertToNPY('../dataset/')


height = 256
width = 256

class Testvideo2tfrecord(unittest.TestCase):
  def test_example1(self):
    n_frames = 5
    convert_videos_to_tfrecord(source_path=in_path, destination_path=out_path,
                               n_videos_in_record=n_videos_per_record,
                               n_frames_per_video=n_frames,
                               dense_optical_flow=True,
                               file_suffix="*.mp4")

    filenames = gfile.Glob(os.path.join(out_path, "*.tfrecords"))
    n_files = len(filenames)

    self.assertTrue(filenames)
    self.assertEqual(n_files * n_videos_per_record,
                     get_number_of_records(filenames, n_frames))

  " travis ressource exhaust, passes locally for 3.6 and 3.4"
  # def test_example2(self):
  #   n_frames = 'all'
  #   convert_videos_to_tfrecord(source_path=in_path, destination_path=out_path,
  #                              n_videos_in_record=n_videos_per_record,
  #                              n_frames_per_video=n_frames,
  #                              n_channels=num_depth, dense_optical_flow=False,
  #                              file_suffix="*.mp4")
  #
  #   filenames = gfile.Glob(os.path.join(out_path, "*.tfrecords"))
  #   n_files = len(filenames)
  #
  #   self.assertTrue(filenames)
  #   self.assertEqual(n_files * n_videos_per_record,
  #                    get_number_of_records(filenames, n_frames))


def read_and_decode(filename_queue, n_frames):
  """Creates one image sequence"""

  reader = tf.TFRecordReader()
  _, serialized_example = reader.read(filename_queue)

  image_seq = []

  if n_frames == 'all':
    n_frames = 354  # travis kills due to too large tfrecord

  for image_count in range(n_frames):
    path = 'blob' + '/' + str(image_count)

    feature_dict = {path: tf.FixedLenFeature([], tf.string),
      'height': tf.FixedLenFeature([], tf.int64),
      'width': tf.FixedLenFeature([], tf.int64),
      'depth': tf.FixedLenFeature([], tf.int64)}

    features = tf.parse_single_example(serialized_example,
                                       features=feature_dict)

    image_buffer = tf.reshape(features[path], shape=[])
    image = tf.decode_raw(image_buffer, tf.uint8)
    image = tf.reshape(image, tf.stack([height, width, num_depth]))
    image = tf.reshape(image, [1, height, width, num_depth])
    image_seq.append(image)

  image_seq = tf.concat(image_seq, 0)

  return image_seq


def get_number_of_records(filenames, n_frames):
  """
  this function determines the number of videos available in all tfrecord files. It also checks on the correct shape of the single examples in the tfrecord
  files.
  :param filenames: a list, each entry containign a (relative) path to one tfrecord file
  :return: the number of overall videos provided in the filenames list
  """

  num_examples = 0

  if n_frames == 'all':
    n_frames_in_test_video = 354
  else:
    n_frames_in_test_video = n_frames

  # create new session to determine batch_size for validation/test data
  with tf.Session() as sess_valid:
    filename_queue_val = tf.train.string_input_producer(filenames, num_epochs=1)
    image_seq_tensor_val = read_and_decode(filename_queue_val, n_frames)

    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())
    sess_valid.run(init_op)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    try:
      while True:
        video = sess_valid.run([image_seq_tensor_val])
        assert np.shape(video) == (1, n_frames_in_test_video, height, width,
                                   num_depth), "shape in the data differs from the expected shape"
        num_examples += 1
    except tf.errors.OutOfRangeError as e:
      coord.request_stop(e)
    finally:
      coord.request_stop()
      coord.join(threads)

  return num_examples


if __name__ == '__main__':
  unittest.main()