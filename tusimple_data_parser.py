import json
import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.misc import imsave

all_datas = []
prev_points = None
points = None


### Check this website for drawing line.
# https://stackoverflow.com/questions/31638651/how-can-i-draw-lines-into-numpy-arrays?rq=1

def trapez(y, y0, w):
    return np.clip(np.minimum(y + 1 + w / 2 - y0, -y + 1 + w / 2 + y0), 0, 1)


def weighted_line(r0, c0, r1, c1, w, rmin=0, rmax=np.inf):
    if abs(c1 - c0) < abs(r1 - r0):
        xx, yy, val = weighted_line(c0, r0, c1, r1, w, rmin=rmin, rmax=rmax)
        return (yy, xx, val)

    if c0 > c1:
        return weighted_line(r1, c1, r0, c0, w, rmin=rmin, rmax=rmax)

    slope = (r1 - r0) / (c1 - c0)

    w *= np.sqrt(1 + np.abs(slope)) / 2

    x = np.arange(c0, c1 + 1, dtype=float)
    y = x * slope + (c1 * r0 - c0 * r1) / (c1 - c0)

    thickness = np.ceil(w / 2)
    yy = (np.floor(y).reshape(-1, 1) + np.arange(-thickness - 1, thickness + 2).reshape(1, -1))
    xx = np.repeat(x, yy.shape[1])
    vals = trapez(yy, y.reshape(-1, 1), w).flatten()

    yy = yy.flatten()

    mask = np.logical_and.reduce((yy >= rmin, yy < rmax, vals > 0))

    return (yy[mask].astype(int), xx[mask].astype(int), vals[mask])


def pre_processing(data_location='./label_data_0313.json', processed_images_location='./image/image{0:04d}.png',
                   processed_labels_location='./label/label{0:04d}.png',
                   line_weights=10,
                   plot_images=False):
    """
    :param data_location: Label data location (ex. './label_data_0313.json')
    :param processed_images_location: You can change images location and name. (ex. './image/image{0:04d}.png')
    :param processed_labels_location: You can change labels location and name. (ex. './label/label{0:04d}.png')
    :param line_weights: Label image line width weight.
    :param plot_images: Plot processed images.
    :return:
    """
    global all_datas, prev_points, points

    with open(data_location) as data_file:
        # print(data_file)
        strlist = data_file.readlines()
        for line in strlist:
            # print(line)
            data = json.loads(line)
            all_datas.append(data)

    all_datas = np.array(all_datas)
    print("total image : {}".format(len(all_datas)))

    for img_n in range(len(all_datas)):
        data = np.zeros([720, 1280])
        lanes = np.array(all_datas[img_n]["lanes"])
        h_samples = np.array(all_datas[img_n]["h_samples"])
        raw_file_name = all_datas[img_n]["raw_file"]

        # print(all_datas[3]["lanes"])

        print("File location : {}".format(raw_file_name))
        for i in range(len(h_samples)):
            for lane_number, lane in enumerate(lanes):  # lanes0 is more than 4 lines
                # print("lane{}".format(lane_number))
                for index, point in enumerate(lane):
                    if not point == -2:
                        points = [point, h_samples[index]]
                        if not prev_points == None:
                            # draw line
                            rr, cc, val = weighted_line(prev_points[1], prev_points[0], points[1], points[0],
                                                        line_weights)
                            data[rr, cc] = 255
                            # data0[points[1], points[0]] = 255
                            # print("prev:{},{}  now:{},{}".format(prev_points[1], prev_points[0], points[1], points[0]))
                        prev_points = points
                        # data0[lane, h_samples0[i]] = 255
                    if index == len(lane) - 1:
                        points = None
                        prev_points = None

        # original data
        img = cv2.cvtColor(cv2.imread(raw_file_name), cv2.COLOR_BGR2RGB)

        if plot_images:
            # created label data show
            plt.imshow(data)
            plt.xticks([]), plt.yticks([])
            plt.show()
            # show original image
            plt.imshow(img)
            plt.xticks([]), plt.yticks([])
            plt.show()

        # save image.
        imsave(processed_images_location.format(img_n), img)
        imsave(processed_labels_location.format(img_n), data)

