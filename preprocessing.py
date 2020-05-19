import numpy as np
import cv2
import os
import tqdm

mp4_path = 'lipread_mp4/'
npy_path = 'lipread_npy_small/'


def ConvertToNPY(data_dir):
    # get all words
    mp4_data = data_dir + mp4_path
    for word in tqdm.tqdm(os.listdir(mp4_data)):
        word_path = data_dir+npy_path + word + '/'
        if not(os.path.isdir(word_path)):
            os.mkdir(word_path)
            for dset in ['train/', 'test/', 'val/']:
                dset_folder = data_dir + mp4_path + word + '/' + dset
                out_folder = data_dir+npy_path + word + '/' + dset
                if not(out_folder):
                    os.mkdir(out_folder)
                for f in os.listdir(dset_folder):
                    if (f.endswith('.mp4')) & (not (os.ispath(out_folder + f.replace('.mp4', '.npy')))):
                        Save(Convert(dset_folder, f, True), dset_folder, f)


def Save(data, path, file):
    # save path
    save_path = path.replace(mp4_path, npy_path) 
    np.save(save_path + file.replace(".mp4", ".npy"), data)


def Convert(path, file, small = True):
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
    # return buf
    return CenterCrop(buf, (96,96))[:, ::2, ::2]
    

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
    # downsample
    small = crop[:, ::2, ::2, :]
    print(small.shape)
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


    cv2.namedWindow('frame 10 - small')
    cv2.imshow('frame 10 - small', small[9])
    cv2.waitKey(0)

    # np.save(path+'.npy', gray)


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





if __name__ == '__main__':
  ConvertToNPY('/mnt/disks/data/dataset/')
  # ConvertTesting('../dataset/lipread_mp4/ABOUT/train/ABOUT_00011')
