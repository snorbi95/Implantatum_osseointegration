import os

import numpy as np
import scipy.interpolate
from scipy import ndimage as ndi
import skimage.color
from skimage import color
from skimage import filters, feature, morphology, measure, segmentation, exposure
from skimage.segmentation import (morphological_chan_vese,
                                  morphological_geodesic_active_contour,
                                  inverse_gaussian_gradient,
                                  checkerboard_level_set, disk_level_set, chan_vese, active_contour)

import cv2
from scipy import ndimage
from sklearn import cluster
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import image as img
from PIL import ImageFilter, Image
from threadpoolctl import threadpool_limits
import imutils
from skimage.filters import rank
from skimage.morphology import disk
from skimage.segmentation import watershed
from skimage.metrics import structural_similarity as ssim
from skimage.transform import rescale
from colorthief import ColorThief

curr_image_name = ''
mask_image = 0
map_image = 0
map_edge_image = 0
gray_image = 0
segmented_rgb = 0
fill_types = {'implant': 255, 'bone': 127, 'other': 192, 'background': 63}
fill_types_rgb = {'implant': [255,255,0], 'bone': [255,0,0], 'other': [0,0,255], 'background': [0,0,0]}
x_coord = 0
y_coord = 0


def delete_small_regions(bin_image, mask_size = 45, step_size = 25):
    for i in range(mask_size // 2, bin_image.shape[0] - mask_size // 2, step_size):
        for j in range(mask_size // 2, bin_image.shape[1] - mask_size // 2, step_size):
            act_mtx = bin_image[i - (mask_size // 2):i + 1 + (mask_size // 2),
                      j - (mask_size // 2):j + 1 + (mask_size // 2)]
            if (np.sum(act_mtx[0,:]) == 0 and np.sum(act_mtx[-1,:]) == 0 and np.sum(act_mtx[:,0]) == 0 and np.sum(act_mtx[:,-1]) == 0)\
                    or act_mtx[act_mtx == 1].size < (mask_size * mask_size) * 0.25:
                act_mtx = np.zeros_like(act_mtx)
            bin_image[i - (mask_size // 2):i + 1 + (mask_size // 2),
            j - (mask_size // 2):j + 1 + (mask_size // 2)] = act_mtx
    return bin_image


def km_clust(array, n_clusters):
    X = array.reshape((-1, 1))
    k_m = cluster.KMeans(n_clusters=n_clusters, n_init=4)
    k_m.fit(X)
    #parameters.wcss.append(k_m.inertia_)
    values = k_m.cluster_centers_.squeeze()
    labels = k_m.labels_
    # silhouette_avg = metrics.silhouette_score(X, labels)
    # sample_silhouette_values = metrics.silhouette_samples(X, labels)

    return(values, labels)

def get_cluster(k,img):
    values, labels = km_clust(img, n_clusters=k)
    res = np.choose(labels, values)
    res.shape, labels.shape = img.shape, img.shape
    #res[np.where(res != res.max())] = 0
    return res


from scipy.spatial import ConvexHull
from PIL import Image, ImageDraw


def smooth_map_image(map_image):
    mask_size = 5
    for i in range(mask_size // 2, map_image.shape[0] - mask_size // 2):
        for j in range(mask_size // 2, map_image.shape[1] - mask_size // 2):
            act_mtx = abs(map_image[i - (mask_size // 2):i + 1 + (mask_size // 2),
                      j - (mask_size // 2):j + 1 + (mask_size // 2)] - map_image[i,j])
            if np.all(act_mtx < 0.25) and np.all(map_image[i - (mask_size // 2):i + 1 + (mask_size // 2), j - (mask_size // 2):j + 1 + (mask_size // 2)] != 0):
                map_image[i - (mask_size // 2):i + 1 + (mask_size // 2),
                j - (mask_size // 2):j + 1 + (mask_size // 2)] = np.average(map_image[i - (mask_size // 2):i + 1 + (mask_size // 2),
                j - (mask_size // 2):j + 1 + (mask_size // 2)])
                print('smoothing...')
    # plt.imshow(map_image)
    # plt.show()
    return map_image

def fill_small_holes():
    mask_size = 5
    for i in range(mask_size // 2, map_image.shape[0] - mask_size // 2, mask_size):
        for j in range(mask_size // 2, map_image.shape[1] - mask_size // 2, mask_size):
            act_mtx = map_image[i - (mask_size // 2):i + 1 + (mask_size // 2),
                          j - (mask_size // 2):j + 1 + (mask_size // 2)]
            act_mtx = np.reshape(act_mtx, (mask_size * mask_size, 1))
            if act_mtx[act_mtx == 255].size > len(act_mtx) * 0.75:
                map_image[i - (mask_size // 2):i + 1 + (mask_size // 2),
                j - (mask_size // 2):j + 1 + (mask_size // 2)] = 255


def flood_fill(coords, type, calls = 0):
    calls += 1
    x = coords[0]
    y = coords[1]
    print(f'Call number: {calls}, coords: [{x}, {y}]')
    if x >= mask_image.shape[0] - 1 or y >= mask_image.shape[1] - 1 or x < 0 or y < 0:
        return
    if mask_image[x,y] == 255 or mask_image[x,y] == 127:
        return
    if map_edge_image[x,y] == 1 or map_image[x,y] == 0:
        return
    # if calls > 1900:
    #     print('Limit reached')
    #     return
    if type == 'implant':
        mask_image[x,y] = 255
    else:
        mask_image[x, y] = 127
    flood_fill([x, y - 1], type, calls)
    flood_fill([x, y + 1], type, calls)
    flood_fill([x - 1, y], type, calls)
    flood_fill([x + 1, y], type, calls)

def flood_fill_iterative(coords, type):
    x = coords[0]
    y = coords[1]
    to_fill = set()
    to_fill.add((x,y))
    while to_fill:
        (x,y) = to_fill.pop()
        if x >= mask_image.shape[0] - 1 or y >= mask_image.shape[1] - 1 or x < 0 or y < 0:
            continue
        elif mask_image[x, y] != 0:
            continue
        elif map_image[x,y] != 1:
            continue
        elif map_edge_image[x,y] > 0.1:
            continue
        mask_image[x, y] = fill_types[type]
        to_fill.add((x - 1, y))
        to_fill.add((x + 1, y))
        to_fill.add((x, y - 1))
        to_fill.add((x, y + 1))


def fill_bone_regions():
    map_flatten = np.reshape(map_edge_image,(map_edge_image.shape[0] * map_edge_image.shape[1],1))
    map_flatten = map_flatten[map_flatten != 0]
    global_avg = np.mean(map_flatten)
    global_std = np.std(map_edge_image)
    gray_flatten = np.reshape(gray_image, (gray_image.shape[0] * gray_image.shape[1], 1))
    gray_flatten = gray_flatten[gray_flatten != 0]
    gray_mean = np.mean(gray_flatten)
    # plt.imshow(gray_image)
    # plt.show()
    mask_size = 17
    for i in range(mask_size // 2, map_edge_image.shape[0] - mask_size // 2,2):
        for j in range(mask_size // 2, map_edge_image.shape[1] - mask_size // 2,2):
            act_mtx = map_edge_image[i - (mask_size // 2):i + 1 + (mask_size // 2),
                      j - (mask_size // 2):j + 1 + (mask_size // 2)]
            act_mtx = np.reshape(act_mtx, (mask_size * mask_size, 1))
            act_gray = gray_image[i - (mask_size // 2):i + 1 + (mask_size // 2),
                      j - (mask_size // 2):j + 1 + (mask_size // 2)]
            act_mtx = act_mtx[act_mtx != 0]
            avg = np.mean(act_mtx)
            if (avg < 0.675) and (np.mean(act_gray) > 0.89 or np.mean(act_gray) < 0.7) and (j < 600 or j > 850 or i > 400):
                #if np.mean(act_gray) > 0.875  and (np.mean(act_gray) > 0.865 or np.mean(act_gray) < 0.675):
                tmp = mask_image[i - (mask_size // 2):i + 1 + (mask_size // 2),
                      j - (mask_size // 2):j + 1 + (mask_size // 2)]
                tmp[tmp == 0] = fill_types['bone']
                mask_image[i - (mask_size // 2):i + 1 + (mask_size // 2),
                j - (mask_size // 2):j + 1 + (mask_size // 2)] = tmp
    # mask_size = 9
    # for i in range(mask_size // 2, map_edge_image.shape[0] - mask_size // 2):
    #     for j in range(mask_size // 2, map_edge_image.shape[1] - mask_size // 2):
    #         act_mtx = map_edge_image[i - (mask_size // 2):i + 1 + (mask_size // 2),
    #                   j - (mask_size // 2):j + 1 + (mask_size // 2)]
    #         act_mtx = np.reshape(act_mtx, (mask_size * mask_size, 1))
    #         expanded_act_mtx = mask_image[i - (mask_size // 2):i + 1 + (mask_size // 2),
    #                   j - (mask_size // 2):j + 1 + (mask_size // 2)]
    #         expanded_act_mtx = np.reshape(expanded_act_mtx, (mask_size * mask_size, 1))
    #         other_num = expanded_act_mtx[expanded_act_mtx == fill_types['other']].size
    #         #act_mtx = act_mtx[act_mtx != 0]
    #         avg = np.mean(act_mtx)
    #         if avg <0.075:
    #             tmp = mask_image[i - (mask_size // 2):i + 1 + (mask_size // 2),
    #                   j - (mask_size // 2):j + 1 + (mask_size // 2)]
    #             tmp[tmp == 0] = fill_types['bone']
    #             mask_image[i - (mask_size // 2):i + 1 + (mask_size // 2),
    #             j - (mask_size // 2):j + 1 + (mask_size // 2)] = tmp
    # mask_size = 15
    # bone_hsv_values = [[0,40],[0,40], [65,100]]
    # # map_edge_image_flatten = np.reshape(map_edge_image, (map_edge_image.shape[0] * map_edge_image.shape[1], 1))
    # # map_edge_image_flatten = map_edge_image_flatten[map_edge_image_flatten != 0]
    # global_avg = np.mean(map_edge_image)
    # # global_med = np.median(map_edge_image_flatten)
    # # global_std = np.std(map_edge_image)
    # # with open('image_features.txt','a') as out_file:
    # #     print(f'{curr_image_name}: Global average - {global_avg}', file=out_file)
    # #     print(f'{curr_image_name}: Global median - {global_med}', file=out_file)
    # for i in range(mask_size // 2, map_edge_image.shape[0] - mask_size // 2):
    #     for j in range(mask_size // 2, map_edge_image.shape[1] - mask_size // 2):
    #         act_mtx = map_edge_image[i - (mask_size // 2):i + 1 + (mask_size // 2),
    #                   j - (mask_size // 2):j + 1 + (mask_size // 2)]
    #         act_mtx = np.reshape(act_mtx, (mask_size * mask_size, 1))
    #         #act_mtx = act_mtx[act_mtx != 0]
    #         avg = np.mean(act_mtx)
    #         hsv_mtx = segmented_hsv[i - (mask_size // 2):i + 1 + (mask_size // 2),
    #                   j - (mask_size // 2):j + 1 + (mask_size // 2)]
    #         channel_h_avg = np.average(hsv_mtx[:,:,0])
    #         channel_s_avg = np.average(hsv_mtx[:, :, 1])
    #         channel_v_avg = np.average(hsv_mtx[:, :, 2])
    #         if bone_hsv_values[0][0] < channel_h_avg < bone_hsv_values[0][1] and\
    #                 bone_hsv_values[1][0] < channel_s_avg < bone_hsv_values[1][1] and bone_hsv_values[2][0] < channel_v_avg < bone_hsv_values[2][1]:
    #             tmp = mask_image[i - (mask_size // 2):i + 1 + (mask_size // 2),
    #                   j - (mask_size // 2):j + 1 + (mask_size // 2)]
    #             tmp[tmp == 0] = fill_types['bone']
    #             mask_image[i - (mask_size // 2):i + 1 + (mask_size // 2),
    #             j - (mask_size // 2):j + 1 + (mask_size // 2)] = tmp

def store_evolution_in(lst):
    """Returns a callback function to store the evolution of the level sets in
    the given list.
    """

    def _store(x):
        lst.append(np.copy(x))

    return _store

def measure_homogenity_of_area(image):
    mask_size = 9
    res_image = np.zeros_like(image)
    for i in range(mask_size // 2, image.shape[0] - mask_size // 2, mask_size // 2):
        for j in range(mask_size // 2, image.shape[1] - mask_size // 2, mask_size // 2):
            act_mtx = image[i - (mask_size // 2):i + 1 + (mask_size // 2),
                      j - (mask_size // 2):j + 1 + (mask_size // 2)]
            act_mtx = np.reshape(act_mtx, (mask_size * mask_size, 1))
            act_mtx = sorted(act_mtx)
            homogenity = np.average(act_mtx[(mask_size * mask_size) // 2:]) - np.average(act_mtx[:(mask_size * mask_size) // 2])
            res_image[i - (mask_size // 2):i + 1 + (mask_size // 2),
                      j - (mask_size // 2):j + 1 + (mask_size // 2)] = homogenity
    plt.imsave('Homogenity images/' + curr_image_name, res_image)

def fill_implant_areas():
    # plt.imshow(mask_image)
    # plt.show()
    # plt.imshow(map_edge_image)
    # plt.show()
    # counts, bins = np.histogram(map_edge_image, bins = 100)
    # counts_gray, bins_gray = np.histogram(gray_image, bins=100)
    # plt.plot(bins[1:], counts, color='r')
    # plt.show()
    map_edge_image_flatten = np.reshape(map_edge_image, (map_edge_image.shape[0] * map_edge_image.shape[1], 1))
    map_edge_image_flatten = map_edge_image_flatten[map_edge_image_flatten != 0]
    global_avg = np.mean(map_edge_image_flatten)
    global_std = np.std(map_edge_image_flatten)
    mask_size = 25
    for i in range(mask_size // 2, map_edge_image.shape[0] - mask_size // 2, 2):
        for j in range(mask_size // 2, map_edge_image.shape[1] - mask_size // 2, 2):
            act_mtx = map_edge_image[i - (mask_size // 2):i + 1 + (mask_size // 2),
                      j - (mask_size // 2):j + 1 + (mask_size // 2)]
            act_gray = gray_image[i - (mask_size // 2):i + 1 + (mask_size // 2),
                      j - (mask_size // 2):j + 1 + (mask_size // 2)]
            act_vec = np.reshape(act_mtx, (mask_size * mask_size, 1))
            # act_gray_vec = np.reshape(act_gray, (mask_size * mask_size, 1))
            # max_gray = np.where(counts_gray[1:] == np.max(counts_gray[1:]))
            num_zeros = act_vec[act_vec < 0.0001].size
            if (np.average(act_vec) < global_avg / 1.5 and np.mean(act_gray) > 0.5) or (i > 600 and i < 800):
            #if np.mean(act_gray) > 0.65:
                tmp = mask_image[i - (mask_size // 2):i + 1 + (mask_size // 2),
                      j - (mask_size // 2):j + 1 + (mask_size // 2)]
                tmp[(tmp == 0)] = fill_types['implant']
                mask_image[i - (mask_size // 2):i + 1 + (mask_size // 2),
                j - (mask_size // 2):j + 1 + (mask_size // 2)] = tmp

from os import listdir
from os.path import isfile, join
import time

image_names = [f for f in listdir('Implantatum-KA_images') if isfile(join('Implantatum-KA_images', f))]

n = 0
for name in image_names:
    print(f'{n}: {name}')
    n += 1

n = 1
sum_time = 0

#1. Process selected image
# image_num = int(input(f'Kerem a beolvasando kep szamat(0 - {len(image_names) - 1}): '))
#
# #1/1. preprocessing of input image
# curr_image_name = image_names[image_num]
# start = time.time()
# rgb_image = img.imread('Implantatum-KA_images/' + curr_image_name)
# image = (color.rgb2gray(rgb_image) * 255).astype(np.uint8)
#
# #1/2. Get edges of the input image
# edge_image = filters.sobel(image)
# plt.imsave('Edge images/' + curr_image_name, edge_image, cmap = 'gray')
# #edge_image = get_binary_image(edge_image)
#
# #1/3. Create contour of implant from the edges
# edge_image = get_binary_image_from_sobel(edge_image)
# edge_image, boundary_coords = get_contour(edge_image)
# segmented_image = image
# #segmented_image[edge_image == 0] = 0
#
# #1/4. Segment contoured area from the input rgb image / Save
# segmented_image = segmented_image[boundary_coords[0]: boundary_coords[1], boundary_coords[2]: boundary_coords[3]]
# bin_segmented_image = np.zeros_like(segmented_image)
# rgb_segmented_image = np.zeros_like(rgb_image)
# coords = np.where(edge_image != 0)
# for i in range(len(coords[0])):
#     rgb_segmented_image[coords[0][i], coords[1][i]] = rgb_image[coords[0][i], coords[1][i]]
# rgb_segmented_image = rgb_segmented_image[boundary_coords[0]: boundary_coords[1], boundary_coords[2]: boundary_coords[3]]
# plt.imshow(rgb_segmented_image)
# plt.show()
# edge_image = edge_image[boundary_coords[0]: boundary_coords[1], boundary_coords[2]: boundary_coords[3]]
#
# #1/5. Crop segmented image with input parameters
# # x = int(input('Kerem a kezdo x koordinátát: '))
# # y = int(input('Kerem a kezdo y koordinatat: '))
# # width = int(input('Kerem a kepszelet szelesseget: '))
# # height = int(input('Kerem a kepszelet magassagat: '))
# # cropped_image = rgb_segmented_image[y: y + height, x: x + width]
# # edge_image = edge_image[y: y + height, x: x + width]
# # print(cropped_image.shape)
#
# hsv_image = color.rgb2hsv(rgb_segmented_image)
#
# # plt.imshow(cropped_image)
# # plt.show()
#
# #1/6. Preprocess cropped image
# fig, ax = plt.subplots(1, 3)
# ax[0].imshow(hsv_image[:,:,0])
# ax[1].imshow(hsv_image[:,:,1])
# ax[2].imshow(hsv_image[:,:,2])
# plt.show()
# gray_image = hsv_image[:,:,2]
# map_edge_image = hsv_image[:,:,1]
#
#
# #1/7. Create edges and thresholded image to separate different areas on the image
# # map_edge_image = morphological_chan_vese(gray_image, 35, init_level_set=init_ls, smoothing=3)
#
# #gray_image = exposure.adjust_log(gray_image, 0.5)
#
# #gray_image = skimage.transform.rescale(gray_image, 0.5, anti_aliasing=False)
# gray_image = exposure.equalize_hist(gray_image)
#
# # cv = chan_vese(gray_image, mu=0.05, lambda1=3, lambda2=1, tol=1e-5, max_iter=200,
# #                dt=0.25, init_level_set="checkerboard", extended_output=True)
# #
# # fig, axes = plt.subplots(2, 2, figsize=(8, 8))
# # ax = axes.flatten()
# #
# # ax[0].imshow(gray_image, cmap="gray")
# # ax[0].set_axis_off()
# # ax[0].set_title("Original Image", fontsize=12)
# #
# # ax[1].imshow(cv[0], cmap="gray")
# # ax[1].set_axis_off()
# # title = "Chan-Vese segmentation - {} iterations".format(len(cv[2]))
# # ax[1].set_title(title, fontsize=12)
# #
# # ax[2].imshow(cv[1], cmap="gray")
# # ax[2].set_axis_off()
# # ax[2].set_title("Final Level Set", fontsize=12)
# #
# # ax[3].plot(cv[2])
# # ax[3].set_title("Evolution of energy over iterations", fontsize=12)
# #
# # fig.tight_layout()
# # plt.show()
# #
# # map_edge_image = filters.sobel(cv[1])
# #
# # plt.imshow(map_edge_image)
# # plt.show()
# #
# # ls = cv[0].astype(np.uint8)
#
# # ls = morphological_chan_vese(gray_image, 100, init_level_set='checkerboard', smoothing=2)
# #
# # fig, axes = plt.subplots(1, 2, figsize=(8, 8))
# # ax = axes.flatten()
# #
# # ax[0].imshow(gray_image, cmap="gray")
# # ax[0].set_axis_off()
# # ax[0].contour(ls, [0.5], colors='r')
# #
# # ax[1].imshow(ls, cmap="gray")
# # ax[1].set_axis_off()
# #
# # plt.show()
#
# ls = morphological_chan_vese(gray_image, 55, init_level_set='checkerboard', smoothing=1)
#
#
# is_invert = input('Invert ls? (y/n): ')
# if is_invert == 'y':
#     ls[ls == 1] = 2
#     ls[ls == 0] = 1
#     ls[ls == 2] = 0
#
#
# fig, ax = plt.subplots(1,2)
# ax[0].imshow(ls, cmap = 'gray')
#
# ls = delete_small_regions(ls)
# ls = morphology.dilation(ls, selem = morphology.square(1))
#
# ax[1].imshow(ls, cmap = 'gray')
# plt.show()
#
# # plt.imshow(gray_image, cmap='gray')
# # plt.show()
#
# fig, axes = plt.subplots(1, 2, figsize=(8, 8))
# ax = axes.flatten()
#
# ax[0].imshow(gray_image, cmap="gray")
# ax[0].set_axis_off()
# ax[0].contour(ls, [0.5], colors='r')
# ax[0].set_title("Morphological ACWE segmentation", fontsize=12)
#
# ax[1].imshow(ls, cmap="gray")
# ax[1].set_axis_off()
# ax[1].legend(loc="upper right")
# title = "Morphological ACWE evolution"
# ax[1].set_title(title, fontsize=12)
#
# plt.show()
#
# gray_image[ls == 1] = 0
#
# # map_edge_ls = measure.find_contours(ls, 0.8)
# # plt.imshow(ls)
# # for contour in map_edge_ls:
# #     plt.plot(contour[:,1], contour[:,0])
# # plt.show()
#
# #Initialize mask image -> Classifier of each pixel on the image
# mask_image = np.zeros((gray_image.shape[0], gray_image.shape[1]))
# mask_image_rgb = np.zeros((gray_image.shape[0], gray_image.shape[1],3))
# #Fill background areas
# mask_image[edge_image == 0] = fill_types['background']
# ls[edge_image == 0] = 0
# mask_image[ls == 1] = fill_types['other']
#
# threshold = filters.threshold_mean(gray_image)
# gray_image_original = np.copy(gray_image)
# gray_image[ls == 1] = 0
#
# map_image = (gray_image > threshold)
# # fd, map_edge_image = feature.hog(gray_image, orientations=4, pixels_per_cell=(16, 16),
# #                      cells_per_block=(4, 4), visualize=True)
# map_edge_images = filters.sobel(gray_image)
# map_edge_image = (map_edge_images + map_edge_image) / 2
# map_edge_image[gray_image == 0] = 0
# plt.imshow(map_edge_image, cmap = 'gray')
# plt.show()
# #map_edge_image[gray_image == 0] = 0
# #fill_small_holes()
#
# fig, ax = plt.subplots(1,4)
# #ax[0].imshow(gray_image_original, cmap='gray')
# ax[0].imshow(gray_image_original, cmap='gray')
# ax[1].imshow(gray_image, cmap='gray')
# ax[2].imshow(map_image, cmap='gray')
# ax[3].imshow(map_edge_image, cmap='gray')
# plt.show()
# plt.imshow(mask_image)
# plt.show()
#
# fill_bone_regions()
#
# plt.imshow(mask_image)
# plt.show()
#
# rgb_segmented_image_b = np.zeros_like(rgb_segmented_image)
# coords = np.where(gray_image != 0)
# for i in range(len(coords[0])):
#     rgb_segmented_image_b[coords[0][i], coords[1][i]] = rgb_segmented_image[coords[0][i], coords[1][i]]
#
# plt.imshow(rgb_segmented_image_b)
# plt.show()
#
# #Fill area using an input starting point with selected type
# while type != 'end':
#     x_coord = int(input('Kerem az x koordinátát: '))
#     y_coord = int(input('Kerem az y koordinátát: '))
#     type = input("Kerem a kitoltes tipusat: ")
#     if type == 'end':
#         break
#     flood_fill_iterative([y_coord, x_coord], 'bone')
#     fig, ax = plt.subplots(1,3)
#     ax[2].scatter([x_coord], [y_coord])
#     ax[0].imshow(gray_image)
#     ax[1].imshow(map_image)
#     ax[2].imshow(mask_image)
#     plt.show()
#
# mask_image[mask_image == 0] = fill_types['implant']
# plt.imshow(mask_image)
# plt.show()
# for key in fill_types_rgb.keys():
#     mask_image_rgb[mask_image == fill_types[key]] = fill_types_rgb[key]
#
# plt.imshow(rgb_segmented_image)
# plt.imshow(mask_image_rgb, cmap = 'jet', alpha = 0.4)
# plt.show()
#
# plt.imsave('Segmented images/' + curr_image_name, rgb_segmented_image)
# plt.imsave('Binary images/' + curr_image_name, edge_image, cmap = 'gray')
# plt.clf()
# print('#' + str(n) + ' Binary image saved')
# end = time.time()
# print(f'Time: {end - start}')
#
# F = mask_image[mask_image != fill_types['background']].size
# M = mask_image[mask_image == fill_types['implant']].size
# B = mask_image[mask_image == fill_types['bone']].size
# A = F - M
# BI = (B / A) * 100
# print(f'Area pixels: {F}\nImplant pixels: {M}\nBone pixels: {B}\nA: {A}\nBI index: {BI}')

n = 1
sum_time = 0

for i in range(39,40):
    image_name = image_names[i]
    type = 'bone'
    print(f'Processing {str(n)} of {str(len(image_names))} images. Image name: {image_name}')
    curr_image_name = image_name
    # image = img.imread('2020-11-02/115_2418_25_7_5x.jpg')
    start = time.time()
    rgb_segmented_image = np.array(img.imread('Segmented images/' + image_name))
    edge_image = color.rgb2gray(img.imread('Binary images/' + image_name))
    hsv_image = color.rgb2hsv(rgb_segmented_image)

    # fig, ax = plt.subplots(1,3)
    # ax[0].imshow(hsv_image[:,:,0])
    # ax[1].imshow(hsv_image[:,:,1])
    # ax[2].imshow(hsv_image[:, :, 2])
    # plt.show()
    # plt.imshow(cropped_image)
    # plt.show()

    # 1/6. Preprocess cropped image
    # fig, ax = plt.subplots(1, 3)
    # ax[0].imshow(hsv_image[:,:,0])
    # ax[1].imshow(hsv_image[:,:,1])
    # ax[2].imshow(hsv_image[:,:,2])
    # plt.show()
    #gray_image = hsv_image[:,:,2]

    # clustered_red = get_cluster(16, rgb_segmented_image[:, :, 0])
    # clustered_green = get_cluster(16, rgb_segmented_image[:, :, 1])
    # clustered_blue = get_cluster(16, rgb_segmented_image[:, :, 2])
    # rgb_segmented_image_for_chan_vese = np.copy(rgb_segmented_image)
    # rgb_segmented_image[:, :, 0] = filters.gaussian(clustered_red, sigma = 2)
    # rgb_segmented_image[:, :, 1] = filters.gaussian(clustered_green, sigma = 2)
    # rgb_segmented_image[:, :, 2] = filters.gaussian(clustered_blue, sigma = 2)
    gray_image = color.rgb2gray(rgb_segmented_image)
    #gray_image = hsv_image[:,:,2]
    #gray_image = filters.gaussian(gray_image, sigma = 2)
    #gray_image = exposure.equalize_hist(gray_image)
    plt.imsave('Gray images/' + curr_image_name, rgb_segmented_image)

    # plt.imshow(rgb_segmented_image_b)
    # plt.show()
    #gray_image = (edge_image + gray_image) ** 2
    counts, bins = np.histogram(gray_image, bins = 100)
    plt.plot(bins[2:], counts[1:], color='r')
    plt.savefig(f'RGB channels/{image_name}')
    plt.clf()
    #edge_image = filters.prewitt(gray_image)
    edge_image = filters.roberts(gray_image)
    #map_edge_image = edge_image ** 2
    map_edge_image = exposure.equalize_hist(edge_image)
    # plt.imshow(map_edge_image)
    # plt.show()
    #map_edge_image = (hsv_image[:, :, 1] + edge_image) / 2
    gray_image = exposure.equalize_hist(gray_image)
    #gray_image = color.rgb2gray(rgb_segmented_image_for_chan_vese)
    #1/7. Create edges and thresholded image to separate different areas on the image
    # map_edge_image = morphological_chan_vese(gray_image, 35, init_level_set=init_ls, smoothing=3)

    #gray_image = exposure.adjust_log(gray_image, 0.5)

    #gray_image = skimage.transform.rescale(gray_image, 0.5, anti_aliasing=False)

    # cv = chan_vese(gray_image, mu=0.05, lambda1=3, lambda2=1, tol=1e-5, max_iter=200,
    #                dt=0.25, init_level_set="checkerboard", extended_output=True)
    #
    # fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    # ax = axes.flatten()
    #
    # ax[0].imshow(gray_image, cmap="gray")
    # ax[0].set_axis_off()
    # ax[0].set_title("Original Image", fontsize=12)
    #
    # ax[1].imshow(cv[0], cmap="gray")
    # ax[1].set_axis_off()
    # title = "Chan-Vese segmentation - {} iterations".format(len(cv[2]))
    # ax[1].set_title(title, fontsize=12)
    #
    # ax[2].imshow(cv[1], cmap="gray")
    # ax[2].set_axis_off()
    # ax[2].set_title("Final Level Set", fontsize=12)
    #
    # ax[3].plot(cv[2])
    # ax[3].set_title("Evolution of energy over iterations", fontsize=12)
    #
    # fig.tight_layout()
    # plt.show()
    #
    # map_edge_image = filters.sobel(cv[1])
    #
    # plt.imshow(map_edge_image)
    # plt.show()
    #
    # ls = cv[0].astype(np.uint8)

    # ls = morphological_chan_vese(gray_image, 100, init_level_set='checkerboard', smoothing=2)
    #
    # fig, axes = plt.subplots(1, 2, figsize=(8, 8))
    # ax = axes.flatten()
    #
    # ax[0].imshow(gray_image, cmap="gray")
    # ax[0].set_axis_off()
    # ax[0].contour(ls, [0.5], colors='r')
    #
    # ax[1].imshow(ls, cmap="gray")
    # ax[1].set_axis_off()
    #
    # plt.show()

    ls = morphological_chan_vese(gray_image, 60, init_level_set='checkerboard', smoothing=1)

    ssim_index = ssim(ls, edge_image)
    #print(ssim_index)
    with open("ssim_indices.txt",'a') as out_file:
        print(f'{image_name}: {ssim_index}', file=out_file)
    if ssim_index > 0.95:
        ls[ls == 1] = 2
        ls[ls == 0] = 1
        ls[ls == 2] = 0
    #
    # fig, ax = plt.subplots(1,2)
    # ax[0].imshow(edge_image, cmap = 'gray')

    ls = delete_small_regions(ls)
    ls = morphology.dilation(ls, selem = morphology.square(1))

    # ax[1].imshow(ls, cmap = 'gray')
    # plt.show()
    #
    # plt.imshow(gray_image, cmap='gray')
    # plt.show()

    fig, axes = plt.subplots(1, 2, figsize=(8, 8))
    ax = axes.flatten()

    ax[0].imshow(gray_image, cmap="gray")
    ax[0].set_axis_off()
    ax[0].contour(ls, [0.5], colors='r')
    ax[0].set_title("Morphological ACWE segmentation", fontsize=12)

    ax[1].imshow(ls, cmap="gray")
    ax[1].set_axis_off()
    ax[1].legend(loc="upper right")
    title = "Morphological ACWE evolution"
    ax[1].set_title(title, fontsize=12)
    plt.savefig(f'Chan Vese Figures/{image_name}')
    plt.clf()

    gray_image[ls == 1] = 0
    map_edge_image[ls == 1] = 0
    map_edge_image = exposure.equalize_hist(map_edge_image)
    gray_image = exposure.equalize_hist(gray_image)
    plt.imsave('Gray images/' + curr_image_name, gray_image, cmap='gray')
    plt.imsave(f'Edges of Segmented Images/{curr_image_name}', map_edge_image)
    rgb_segmented_image_b = np.zeros_like(rgb_segmented_image)
    coords = np.where(gray_image != 0)
    for i in range(len(coords[0])):
        rgb_segmented_image_b[coords[0][i], coords[1][i]] = rgb_segmented_image[coords[0][i], coords[1][i]]
    segmented_hsv = (color.rgb2hsv(rgb_segmented_image_b) * 255).astype(np.uint8)

    # fig, ax = plt.subplots(1, 3)
    # ax[0].imshow(segmented_hsv[:,:,0])
    # ax[1].imshow(segmented_hsv[:,:,1])
    # ax[2].imshow(segmented_hsv[:,:,2])
    # plt.show()
    # map_edge_ls = measure.find_contours(ls, 0.8)
    # plt.imshow(ls)
    # for contour in map_edge_ls:
    #     plt.plot(contour[:,1], contour[:,0])
    # plt.show()

    #Initialize mask image -> Classifier of each pixel on the image
    mask_image = np.zeros((gray_image.shape[0], gray_image.shape[1]))
    mask_image_rgb = np.zeros((gray_image.shape[0], gray_image.shape[1],3))
    #Fill background areas
    mask_image[edge_image == 0] = fill_types['background']
    ls[edge_image == 0] = 0
    mask_image[ls == 1] = fill_types['other']

    gray_image_original = np.copy(gray_image)

    threshold = filters.threshold_mean(gray_image)

    map_image = (gray_image > threshold)
    # counts_h, bins_h = np.histogram(map_edge_images, bins=256)
    # max_h = np.where(counts_h[1:] == np.max(counts_h[1:]))
    # max_h = max_h[0] / 256
    # with open('image_features.txt','a') as out_file:
    #     print(f'{curr_image_name}: Maximum value: {max_h}', file=out_file)
    # if max_h < 0.075:
    #     gray_image[np.abs(map_edge_images - max_h) > 0.1] = 0
    # else:
    #     gray_image[np.abs(map_edge_images - max_h) < 0.1] = 0
    # fig, ax = plt.subplots(1,2)
    # ax[0].plot(bins_h[2:], counts_h[1:], color='r')
    # ax[1].imshow(gray_image)
    # plt.savefig(f'Edges of Segmented Images/{curr_image_name}')
    # plt.clf()
    # plt.imshow(gray_image)
    # plt.show()
    # fd, map_edge_image = feature.hog(gray_image, orientations=4, pixels_per_cell=(16, 16),
    #                      cells_per_block=(4, 4), visualize=True)
    # plt.imshow(map_edge_images)
    # plt.show()
    # fig, ax = plt.subplots(1,2)
    # ax[0].imshow(map_edge_images, cmap='gray')
    # counts, bins = np.histogram(map_edge_images, bins = 256)
    # ax[1].plot(bins[2:], counts[1:], color='r')
    # plt.show()
    # plt.imshow(map_edge_image)
    # plt.show()
    #map_edge_image = map_edge_images
    # counts, bins = np.histogram(map_edge_image, bins=256)
    # fig, ax = plt.subplots(1,2)
    # ax[0].imshow(map_edge_image)
    # ax[1].bar(bins[1:], counts)
    # plt.show()
    # plt.imshow(map_edge_image, cmap = 'gray')
    # plt.show()
    #map_edge_image[gray_image == 0] = 0
    #fill_small_holes()
    plt.clf()
    #fig, ax = plt.subplots(1,4)
    #ax[0].imshow(gray_image_original, cmap='gray')
    # ax[0].imshow(gray_image_original, cmap='gray')
    # ax[1].imshow(gray_image, cmap='gray')
    # ax[2].imshow(map_image, cmap='gray')
    # ax[3].imshow(map_edge_image, cmap='gray')
    # plt.show()
    # plt.imshow(mask_image)
    # plt.show()
    segmented_rgb = rgb_segmented_image_b
    if type == 'implant':
        fill_implant_areas()
        mask_image[mask_image == 0] = fill_types['bone']
    else:
        fill_bone_regions()
        mask_image[mask_image == 0] = fill_types['implant']

    # plt.imshow(mask_image)
    # plt.show()

    # plt.imshow(rgb_segmented_image_b)

    # plt.imsave('RGB of Segmented Images/' + curr_image_name + '_channel1.jpg', rgb_segmented_image_b[:,:,0])
    # plt.imsave('RGB of Segmented Images/' + curr_image_name + '_channel2.jpg', rgb_segmented_image_b[:, :, 1])
    # plt.imsave('RGB of Segmented Images/' + curr_image_name + '_channel3.jpg', rgb_segmented_image_b[:, :, 2])
    # counts_red, bins_red = np.histogram(rgb_segmented_image_b[:,:,0], bins = 256)
    # max_red = np.where(counts_red[1:] == np.max(counts_red[1:]))
    # counts_green, bins_green = np.histogram(rgb_segmented_image_b[:, :, 1], bins=256)
    # max_green = np.where(counts_green[1:] == np.max(counts_green[1:]))
    # counts_blue, bins_blue = np.histogram(rgb_segmented_image_b[:, :, 2], bins=256)
    # max_blue = np.where(counts_blue[1:] == np.max(counts_blue[1:]))
    # r_channel = rgb_segmented_image_b[:,:,0]
    # r_channel[np.abs(r_channel - max_red) < 25] = 0
    # g_channel = rgb_segmented_image_b[:, :, 1]
    # g_channel[np.abs(r_channel - max_green) < 25] = 0
    # b_channel = rgb_segmented_image_b[:, :, 2]
    # b_channel[np.abs(r_channel - max_blue) < 25] = 0
    # rgb_segmented_image_b[:,:,0] = r_channel
    # rgb_segmented_image_b[:,:,1] = g_channel
    # rgb_segmented_image_b[:,:,2] = b_channel
    #
    #
    # plt.plot(bins_red[2:], counts_red[1:], color='r')
    # plt.plot(bins_green[2:], counts_green[1:], color='g')
    # plt.plot(bins_blue[2:], counts_blue[1:], color='b')
    # plt.savefig(f'RGB channels/{image_name}')
    # plt.clf()
    # hsv_segmented_image = color.rgb2hsv(rgb_segmented_image_b)
    # plt.imshow(hsv_segmented_image[:,:,1] * hsv_segmented_image[:,:,2])
    # plt.show()
    # plt.imsave('HSV of Segmented Images/' + curr_image_name + '_channel1.jpg', hsv_segmented_image[:, :, 0])
    # plt.imsave('HSV of Segmented Images/' + curr_image_name + '_channel2.jpg', hsv_segmented_image[:, :, 1])
    # plt.imsave('HSV of Segmented Images/' + curr_image_name + '_channel3.jpg', hsv_segmented_image[:, :, 2])
    #
    # counts_h, bins_h = np.histogram(hsv_segmented_image[:, :, 0], bins=256)
    # counts_s, bins_s = np.histogram(hsv_segmented_image[:, :, 1], bins=256)
    # counts_v, bins_v = np.histogram(hsv_segmented_image[:, :, 2], bins=256)
    #
    # plt.plot(bins_h[10:], counts_h[9:], color='r')
    # plt.plot(bins_s[10:], counts_s[9:], color='g')
    # plt.plot(bins_v[10:], counts_v[9:], color='b')
    # plt.savefig(f'HSV channels/{image_name}')
    # plt.clf()

    # plt.imshow(mask_image)
    # plt.show()
    for key in fill_types_rgb.keys():
        mask_image_rgb[mask_image == fill_types[key]] = fill_types_rgb[key]

    # plt.imshow(rgb_segmented_image)
    # plt.imshow(mask_image_rgb, cmap = 'jet', alpha = 0.4)
    # plt.show()
    F = mask_image[mask_image != fill_types['background']].size
    M = mask_image[mask_image == fill_types['implant']].size
    B = mask_image[mask_image == fill_types['bone']].size
    A = F - M
    BI = (B / A) * 100
    print(f'Area pixels: {F}\nImplant pixels: {M}\nBone pixels: {B}\nA: {A}\nBI index: {BI}')
    with open("bi_indices.txt",'a') as out_file:
        print(f'{image_name}: {BI}%', file=out_file)
    plt.imshow(rgb_segmented_image)
    plt.imshow(mask_image_rgb, cmap = 'jet', alpha = 0.25)
    plt.savefig(f'Labelled images/{image_name}')
    plt.clf()
    print('#' + str(n) + ' Binary image saved')
    end = time.time()
    print(f'Time: {end - start}')
    sum_time += end - start
    n += 1

print(f'Average time: {sum_time / (n - 1)}')