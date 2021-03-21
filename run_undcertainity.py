import os

import numpy as np
import pandas as pd
import pydicom as dicom

from keras import backend as K
from keras.models import load_model

from matplotlib import pylab as plt

from tqdm import tqdm
from scipy.misc import imresize
from skimage.measure import moments
from scipy.ndimage.interpolation import rotate


RED_COLOR = "#ff0000"
YELLOW_COLOR = "#FFFF00"
TEAL_COLOR = "#00ffff"

NORM_VECT_X = np.array([5, -1, 0])
NORM_VECT_Y = np.array([-1, 2, -0.5])

ROTATION_LST = np.arange(-6, 6 + 1) * 10
ROTATION_LST = [x for x in ROTATION_LST if x != 0]

LANDMARK_LST = ["Apex", "MV"]

# functions
def preproces_img(mtx, resize_target, orient=None):
    """
    INPUT:
        mtx:
            the input numpy matrix of dicom files
        resize_target:
            tuple of Y-X size
        curr_fov:
            the field of view for the dicom series
        target_fov:
            the field of view for the target
        orientation:
            the current orientation tuple of the series
    OUTPUT:
        the processed matrix
    """
    # try to standardize orientation
    # get orientation for row and cols
    orient_row = [float(orient[0]),float(orient[1]),float(orient[2])]
    orient_col = [float(orient[3]),float(orient[4]),float(orient[5])]

    # determine if we flip left/right
    curr_mtx = mtx.copy()
    if np.dot(orient_row, NORM_VECT_X) < 0:
        curr_mtx = np.flip(curr_mtx, axis=1)
    # determine if we flip up/down
    if np.dot(orient_col, NORM_VECT_Y) < 0:
        curr_mtx = np.flip(curr_mtx, axis=0)


    # resize
    curr_mtx = imresize(curr_mtx, resize_target)


    return curr_mtx

def image_2nd_moment(pred_data):
    # https://en.wikipedia.org/wiki/Image_moment

    # calculate moments
    M = moments(pred_data)

    # centroids
    x_bar = M[1, 0]/ M[0, 0]
    y_bar = M[0, 1]/ M[0, 0]

    # moments
    mu_prime_20 = M[2, 0]/M[0, 0] - x_bar**2
    mu_prime_02 = M[0, 2]/M[0, 0] - y_bar**2
    mu_prime_11 = M[1, 1]/M[0, 0] - x_bar * y_bar

    # calculate
    return np.sqrt((mu_prime_02**2 + mu_prime_20**2))

def plot_img(image, overlay, save_path):
    # create subplot
    fig, ax = plt.subplots(1, sharey = True, dpi = 125)

    # turn off ticks
    ax.axis('off')

    # plots
    plt.imshow(image, "gray")
    plt.imshow(overlay, "inferno", alpha = 0.3)

    # save and close
    fig.savefig(save_path)
    plt.close(fig)

def split_all(path):
    # adapted from: https://www.oreilly.com/library/view/python-cookbook/0596001673/ch04s16.html
    all_parts = []
    while 1:
        parts = os.path.split(path)
        if parts[0] == path:  # sentinel for absolute paths
            all_parts.insert(0, parts[0])
            break
        elif parts[1] == path: # sentinel for relative paths
            all_parts.insert(0, parts[1])
            break
        else:
            path = parts[0]
            all_parts.insert(0, parts[1])
    return all_parts




# HACK
data_path = "data/"

# HACK
output_path = "output/"



os.listdir(data_path)


# list all dicom files
rslt_f_lst = []
for dir_name, sub_dir, f_lst in os.walk(data_path):
    for f_name in f_lst:
        rslt_f_lst.append(os.path.join(dir_name, f_name))

# remove non dicom files
rslt_f_lst = [x for x in rslt_f_lst if ".dcm" in x]


# load model
# HACK
model_path = "model/iter_1_model_2.hdf5"
model = load_model(model_path)

# take last two elements as a unique id
# HACK


# iniitalize results df
rslt_df = pd.DataFrame()


# iterate over each dcm
for curr_f in tqdm(rslt_f_lst):
    curr_f = rslt_f_lst[0]
    curr_dcm = dicom.read_file(curr_f)
    curr_img = curr_dcm.pixel_array


    # apply preprocessing
    curr_img = preproces_img(
        curr_img, # image mtx
        (128, 128), # resize target
        curr_dcm.ImageOrientationPatient, # image orientation
    )


    # predict for base prediction matrix
    base_pred = model.predict(np.expand_dims(np.expand_dims(curr_img, axis=0), axis=-1))[0]


    # get rotation data
    # rotation augmentations
    rot_pred_dict = {}
    for curr_angle in ROTATION_LST:
        tmp_array = rotate(curr_img, curr_angle, reshape=False)
        tmp_array = np.expand_dims(tmp_array, axis=-1)
        tmp_array = np.expand_dims(tmp_array, axis=0)
        tmp_pred = model.predict(tmp_array)[0]
        rot_pred_dict[curr_angle] = rotate(tmp_pred, -curr_angle, reshape=False)

    # rotation data
    org_rot_lst = []
    rot_rot_lst = []
    for curr_angle in ROTATION_LST:
        # rotate and unrotate org heatmap
        org_1 = rotate(base_pred, curr_angle, reshape=False)
        org_2 = rotate(org_1, -curr_angle, reshape=False)

        # get rotated
        rot = rot_pred_dict[curr_angle]

        # normalize by channel
        org_2[:, :, 0] = (org_2[:, :, 0] - org_2[:, :, 0].min())
        org_2[:, :, 1] = (org_2[:, :, 1] - org_2[:, :, 1].min())

        org_2[:, :, 0] = (org_2[:, :, 0] / org_2[:, :, 0].sum())
        org_2[:, :, 1] = (org_2[:, :, 1] / org_2[:, :, 1].sum())

        rot[:, :, 0] = (rot[:, :, 0] - rot[:, :, 0].min())
        rot[:, :, 1] = (rot[:, :, 1] - rot[:, :, 1].min())

        rot[:, :, 0] = (rot[:, :, 0] / rot[:, :, 0].sum())
        rot[:, :, 1] = (rot[:, :, 1] / rot[:, :, 1].sum())

        # add to lists
        org_rot_lst.append(org_2)
        rot_rot_lst.append(rot)

    # calculate rotational entropy
    abs_diff_rot_map = np.array(org_rot_lst) - np.array(rot_rot_lst)
    abs_diff_rot_map = np.abs(abs_diff_rot_map)
    abs_diff_rot_map = abs_diff_rot_map.sum(axis=0)


    # iterate over differenet landmarks
    for a_indx, anat in enumerate(LANDMARK_LST):
        # get uncertainity measurements
        psuedoprob_max = base_pred[:, :, a_indx].max()
        rotational_entropy = image_2nd_moment(abs_diff_rot_map[:, :, a_indx])


        # construct save path
        save_path = "_".join(split_all(curr_f)[-2:])
        save_path = save_path.replace(".dcm", "")

        pseudoprob_save_path = anat + "_pseudo_prob_" + save_path
        pseudoprob_save_path = os.path.join(output_path, pseudoprob_save_path)
        plot_img(curr_img, base_pred[:, :, a_indx], pseudoprob_save_path)

        rot_ent_save_path = anat + "_rot_ent_" + save_path
        rot_ent_save_path = os.path.join(output_path, rot_ent_save_path)
        plot_img(curr_img, abs_diff_rot_map[:, :, a_indx], rot_ent_save_path)


        # add results to data frame
        rslt_df = rslt_df.append(pd.DataFrame({
            # uncertainity metrics
            "psuedoprob_max": [psuedoprob_max],
            "rotational_entropy": [rotational_entropy],

            # id info
            "id": [save_path],
            "landmark": [anat],
        })).reset_index(drop=True)
