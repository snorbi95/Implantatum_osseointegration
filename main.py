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

curr_image_name = ''
mask_image = 0
map_image = 0
map_edge_image = 0
fill_types = {'implant': 255, 'bone': 127, 'other': 192, 'background': 63}
fill_types_rgb = {'implant': [255,255,0], 'bone': [255,0,0], 'other': [0,0,255], 'background': [0,0,0]}
x_coord = 0
y_coord = 0


def get_binary_image(edges):
    mask_size = 25
    res = np.zeros_like(edges)
    for i in range(mask_size // 2, edges.shape[0] - mask_size // 2 , mask_size):
        for j in range(mask_size // 2, edges.shape[1] - mask_size // 2, mask_size):
            act_mtx = edges[i - (mask_size // 2):i + 1 + (mask_size // 2),
                          j - (mask_size // 2):j + 1 + (mask_size // 2)]
            act_mtx = np.reshape(act_mtx, (mask_size * mask_size, 1))
            trues = act_mtx[act_mtx == True].size
            if trues > (mask_size * mask_size) * 0.25:
                res[i - (mask_size // 2):i + 1 + (mask_size // 2),
                j - (mask_size // 2):j + 1 + (mask_size // 2)] = 1
            # else:
            #     edges[i - (mask_size // 2):i + 1 + (mask_size // 2),
            #     j - (mask_size // 2):j + 1 + (mask_size // 2)] = 0

    res = morphology.opening(res, selem=morphology.square(25))
    # plt.imshow(res)
    # plt.show()
    return res

def delete_small_regions(bin_image, mask_size = 25, step_size = 1):
    for i in range(mask_size // 2, bin_image.shape[0] - mask_size // 2, step_size):
        for j in range(mask_size // 2, bin_image.shape[1] - mask_size // 2, step_size):
            act_mtx = bin_image[i - (mask_size // 2):i + 1 + (mask_size // 2),
                      j - (mask_size // 2):j + 1 + (mask_size // 2)]
            if np.sum(act_mtx[0,:]) == 0 and np.sum(act_mtx[-1,:]) == 0 and np.sum(act_mtx[:,0]) == 0 and np.sum(act_mtx[:,-1]) == 0:
                act_mtx = 0
            bin_image[i - (mask_size // 2):i + 1 + (mask_size // 2),
            j - (mask_size // 2):j + 1 + (mask_size // 2)] = act_mtx
    return bin_image

def get_binary_image_from_sobel(edges, mask_size, step_size):
    print('Binary image....')
    res = np.zeros_like(edges)
    avg = np.mean(edges)
    print(f'Average edge size: {avg}')
    for i in range(mask_size // 2, edges.shape[0] - mask_size // 2, step_size):
        for j in range(mask_size // 2, edges.shape[1] - mask_size // 2, step_size):
            act_mtx = edges[i - (mask_size // 2):i + 1 + (mask_size // 2),
                          j - (mask_size // 2):j + 1 + (mask_size // 2)]
            act_mtx = np.reshape(act_mtx, (mask_size * mask_size, 1))
            above_avg = act_mtx[act_mtx > avg].size
            if above_avg > (mask_size * mask_size) * 0.65:
                res[i - (mask_size // 2):i + 1 + (mask_size // 2),
                j - (mask_size // 2):j + 1 + (mask_size // 2)] = 1
            # else:
            #     edges[i - (mask_size // 2):i + 1 + (mask_size // 2),
            #     j - (mask_size // 2):j + 1 + (mask_size // 2)] = 0

    # plt.imshow(res)
    # plt.show()
    res = morphology.closing(res, selem=morphology.square(mask_size))
    #res = delete_small_regions(res, mask_size = 75, step_size=35)
    #res = morphology.erosion(res, selem=morphology.square(mask_size))
    # plt.imshow(res)
    # plt.show()
    return res

def get_max_len_list(li):
    # max_len = 0
    # max_list = []
    # for item in li:
    #     if item.shape[0] > max_len:
    #         max_len = item.shape[0]
    #         max_list = item
    # return max_list
    len_and_list = [(item.shape[0], item) for item in li]
    len_and_list = sorted(len_and_list, key=lambda x:x[0], reverse=True)
    return len_and_list[0]

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

def get_contour(bin_image):
    print('Contour image....')
    contours = measure.find_contours(bin_image, level=0.8, fully_connected='low')
    # fig, ax = plt.subplots()
    bin_image = ndimage.binary_fill_holes(bin_image)
    #plt.imshow(bin_image, cmap=plt.cm.gray)
    #print(contours)
    max_contour = get_max_len_list(contours)
    #plt.plot(contours[:, 1], contours[:, 0], linewidth=2)
    #plt.show()
    res_image = np.zeros_like(bin_image)
    contour = max_contour[1].astype(np.uint32)
    res_image[contour[:,0], contour[:,1]] = 255
    res_image = ndimage.binary_fill_holes(res_image)
    #plt.imshow(res_image)
    # plt.show()
    seeds = (np.where(res_image == 1)[0], np.where(res_image == 1)[1])
    start_x = np.min(seeds[0])
    start_y = np.min(seeds[1])
    end_x = np.max(seeds[0])
    end_y = np.max(seeds[1])
    for contour in contours:
        if np.max(contour[:,0]) < end_x and np.min(contour[:,0]) > start_x and np.max(contour[:,1]) < end_y and np.min(contour[:,1]) > start_y and len(contour[0]) > 50:
            contour = contour.astype(np.uint32)
            #plt.plot(contour[:, 1], contour[:, 0], linewidth=2)
            res_image[contour[:, 0], contour[:, 1]] = 255
    #res_image = ndimage.binary_fill_holes(res_image)
    res_image = morphology.closing(res_image, selem = morphology.square(35))
    res_image = ndimage.binary_fill_holes(res_image)
    seeds = (np.where(res_image == 1)[0], np.where(res_image == 1)[1])
    #res_image = morphology.convex_hull_image(res_image)
    start_x = np.min(seeds[0])
    start_y = np.min(seeds[1])
    end_x = np.max(seeds[0])
    end_y = np.max(seeds[1])
    # res_image[start_x:end_x, start_y: end_y] = bin_image[start_x:end_x, start_y: end_y]
    # plt.imshow(bin_image)
    # plt.show()
    # plt.imshow(points_image)
    # plt.show()
    #res_image = morphology.flood(res_image, seed)
    #res_image = morphology.convex_hull_image(res_image)
    #plt.imsave('Points images/' + image_name, points_image, cmap='gray')
    # ax.axis('image')
    # ax.set_xticks([])
    # ax.set_yticks([])
    # res_image = morphology.convex_hull_image(res_image)
    # plt.imshow(res_image)
    # plt.show()

    return res_image, [start_x, end_x, start_y, end_y]

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
    mask_size = 15
    for i in range(mask_size // 2, map_edge_image.shape[0] - mask_size // 2):
        for j in range(mask_size // 2, map_edge_image.shape[1] - mask_size // 2):
            act_mtx = map_edge_image[i - (mask_size // 2):i + 1 + (mask_size // 2),
                      j - (mask_size // 2):j + 1 + (mask_size // 2)]
            act_mtx = np.reshape(act_mtx, (mask_size * mask_size, 1))
            act_mtx = act_mtx[act_mtx != 0]
            avg = np.mean(act_mtx)
            if avg < 0.075 or avg > 0.25:
                tmp = mask_image[i - (mask_size // 2):i + 1 + (mask_size // 2),
                      j - (mask_size // 2):j + 1 + (mask_size // 2)]
                tmp[tmp == 0] = fill_types['bone']
                mask_image[i - (mask_size // 2):i + 1 + (mask_size // 2),
                j - (mask_size // 2):j + 1 + (mask_size // 2)] = tmp

def store_evolution_in(lst):
    """Returns a callback function to store the evolution of the level sets in
    the given list.
    """

    def _store(x):
        lst.append(np.copy(x))

    return _store

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

#2. Process every image
# for image_name in image_names:
#     type = ''
#     print(f'Processing {str(n)} of {str(len(image_names))} images. Image name: {image_name}')
#     curr_image_name = image_name
#     #image = img.imread('2020-11-02/115_2418_25_7_5x.jpg')
#     start = time.time()
#     rgb_image = img.imread('Implantatum-KA_images/' + image_name)
#     hsv_image = color.rgb2hsv(rgb_image)
#     # fig, ax = plt.subplots(1,3)
#     # ax[0].imshow(hsv_image[:, :, 0])
#     # ax[1].imshow(hsv_image[:, :, 1])
#     # ax[2].imshow(hsv_image[:, :, 2])
#     # plt.show()
#     threshold = filters.threshold_mean(hsv_image[:,:,0])
#     gray_image = hsv_image[:,:,2]
#     gray_image[hsv_image[:,:,0] > threshold] = 0
#     # plt.imshow(gray_image)
#     # plt.show()
#     # plt.imshow((gray_image_2 + gray_image) / 2)
#     # plt.show()
#
#     image = (color.rgb2gray(rgb_image) * 255).astype(np.uint8)
#     #edge_image = feature.canny(edge_image, sigma=1)
#     edge_image = filters.sobel(gray_image)
#     #edge_image = morphology.closing(edge_image, selem=morphology.square(5))
#     # plt.imshow(edge_image)
#     # plt.show()
#     plt.imsave('Edge images/' + image_name, edge_image, cmap = 'gray')
#     #edge_image = get_binary_image(edge_image)
#     edge_image = get_binary_image_from_sobel(edge_image, mask_size=15, step_size=15)
#     edge_image, boundary_coords = get_contour(edge_image)
#     segmented_image = image
#     #segmented_image[edge_image == 0] = 0
#     segmented_image = segmented_image[boundary_coords[0]: boundary_coords[1], boundary_coords[2]: boundary_coords[3]]
#     mask_size = 196
#     bin_segmented_image = np.zeros_like(segmented_image)
#             #bin_segmented_image[i:i + mask_size, j: j + mask_size] = temp_image
#             # plt.imshow(cluster_image)
#             # plt.show()
#     # plt.imshow(bin_segmented_image)
#     # plt.show()
#     #segmented_image[edge_image == 0] = 0
#     rgb_segmented_image = np.zeros_like(rgb_image)
#     coords = np.where(edge_image != 0)
#     for i in range(len(coords[0])):
#         rgb_segmented_image[coords[0][i], coords[1][i]] = rgb_image[coords[0][i], coords[1][i]]
#     rgb_segmented_image = rgb_segmented_image[boundary_coords[0]: boundary_coords[1], boundary_coords[2]: boundary_coords[3]]
#     edge_image = edge_image[boundary_coords[0]: boundary_coords[1], boundary_coords[2]: boundary_coords[3]]
#     # plt.imshow(rgb_segmented_image)
#     # plt.show()
#     # patch_n = 0
#     # for i in range(0, rgb_segmented_image.shape[0], mask_size):
#     #     for j in range(0, rgb_segmented_image.shape[1], mask_size):
#     #         temp_image = rgb_segmented_image[i:i + mask_size, j: j + mask_size]
#     #         if temp_image.shape[0] < mask_size or temp_image.shape[1] < mask_size:
#     #             temp_zeros = np.zeros((mask_size, mask_size, 3))
#     #             temp_zeros[:temp_image.shape[0], :temp_image.shape[1]] = temp_image
#     #             temp_image = temp_zeros.astype(np.uint8)
#     #         print(temp_image.shape)
#     #         hsv_image = color.rgb2hsv(temp_image)
#     #         # fig, ax = plt.subplots(1,3)
#     #         # ax[0].imshow(hsv_image[:, :, 0])
#     #         # ax[1].imshow(hsv_image[:, :, 1])
#     #         # ax[2].imshow(hsv_image[:, :, 2])
#     #         # plt.show()
#     #         map_image = hsv_image[:, :, 2]
#     #         map_image = smooth_map_image(map_image)
#     #         patch_n += 1
#     #         mask_image = np.zeros((temp_image.shape[0], temp_image.shape[1]))
#     #         current_coords = [100,60]
#     #         temp_edge_image = color.rgb2gray(filters.sobel(map_image))
#     #         mask_image = snake_fill(mask_image, temp_edge_image, map_image, current_coords)
#     #         fig, ax = plt.subplots(1,3)
#     #         ax[0].imshow(map_image)
#     #         ax[1].imshow(temp_edge_image)
#     #         ax[2].imshow(mask_image)
#     #         plt.show()
#             # try:
#             #     plt.imsave(f'Image patches/{image_name}/{image_name}patch_{i}-{j}.png', temp_image)
#             # except:
#             #     os.mkdir('Image patches/' + image_name)
#             #     plt.imsave(f'Image patches/{image_name}/{image_name}patch_{i}-{j}.png', temp_image)
#     # rgb_segmented_image = rgb_segmented_image[boundary_coords[0]: boundary_coords[1], boundary_coords[2]: boundary_coords[3]]
#
#     # plt.imshow(segmented_image)
#     # plt.show()
#     #get_implanted_area(rgb_segmented_image)
#     # plt.show()
#     #plt.imshow(edge_image)
#     #image[edge_image == 0] //= 3
#     plt.imsave('Segmented images/' + image_name, rgb_segmented_image)
#     plt.imsave('Binary images/' + image_name, edge_image, cmap = 'gray')
#     plt.clf()
#     print('#' + str(n) + ' Binary image saved')
#     end = time.time()
#     print(f'Time: {end - start}')
#     sum_time += end - start
#     n += 1

n = 1
sum_time = 0

for image_name in image_names:
    type = ''
    print(f'Processing {str(n)} of {str(len(image_names))} images. Image name: {image_name}')
    curr_image_name = image_name
    # image = img.imread('2020-11-02/115_2418_25_7_5x.jpg')
    start = time.time()
    rgb_segmented_image = img.imread('Segmented images/' + image_name)
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
    gray_image = hsv_image[:,:,2]
    map_edge_image = hsv_image[:,:,1]


    #1/7. Create edges and thresholded image to separate different areas on the image
    # map_edge_image = morphological_chan_vese(gray_image, 35, init_level_set=init_ls, smoothing=3)

    #gray_image = exposure.adjust_log(gray_image, 0.5)

    #gray_image = skimage.transform.rescale(gray_image, 0.5, anti_aliasing=False)
    gray_image = exposure.equalize_hist(gray_image)

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

    ls = morphological_chan_vese(gray_image, 55, init_level_set='checkerboard', smoothing=2)

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

    threshold = filters.threshold_mean(gray_image)
    gray_image_original = np.copy(gray_image)
    gray_image[ls == 1] = 0

    map_image = (gray_image > threshold)
    # fd, map_edge_image = feature.hog(gray_image, orientations=4, pixels_per_cell=(16, 16),
    #                      cells_per_block=(4, 4), visualize=True)
    map_edge_images = filters.sobel(gray_image)
    map_edge_image = (map_edge_images + map_edge_image) / 2
    map_edge_image[gray_image == 0] = 0
    # counts, bins = np.histogram(map_edge_image, bins=256)
    # fig, ax = plt.subplots(1,2)
    # ax[0].imshow(map_edge_image)
    # ax[1].bar(bins[1:], counts)
    # plt.show()
    plt.imsave('Edges of Segmented Images/' + curr_image_name, map_edge_image)
    # plt.imshow(map_edge_image, cmap = 'gray')
    # plt.show()
    #map_edge_image[gray_image == 0] = 0
    #fill_small_holes()

    # fig, ax = plt.subplots(1,4)
    # #ax[0].imshow(gray_image_original, cmap='gray')
    # ax[0].imshow(gray_image_original, cmap='gray')
    # ax[1].imshow(gray_image, cmap='gray')
    # ax[2].imshow(map_image, cmap='gray')
    # ax[3].imshow(map_edge_image, cmap='gray')
    # plt.show()
    # plt.imshow(mask_image)
    # plt.show()

    fill_bone_regions()

    # plt.imshow(mask_image)
    # plt.show()

    rgb_segmented_image_b = np.zeros_like(rgb_segmented_image)
    coords = np.where(gray_image != 0)
    for i in range(len(coords[0])):
        rgb_segmented_image_b[coords[0][i], coords[1][i]] = rgb_segmented_image[coords[0][i], coords[1][i]]

    # plt.imshow(rgb_segmented_image_b)
    plt.imsave('Gray images/' + curr_image_name, rgb_segmented_image_b)
    plt.imsave('RGB of Segmented Images/' + curr_image_name + '_channel1.jpg', rgb_segmented_image_b[:,:,0])
    plt.imsave('RGB of Segmented Images/' + curr_image_name + '_channel2.jpg', rgb_segmented_image_b[:, :, 1])
    plt.imsave('RGB of Segmented Images/' + curr_image_name + '_channel3.jpg', rgb_segmented_image_b[:, :, 2])
    counts_red, bins_red = np.histogram(rgb_segmented_image_b[:,:,0], bins = 256)
    counts_green, bins_green = np.histogram(rgb_segmented_image_b[:, :, 1], bins=256)
    counts_blue, bins_blue = np.histogram(rgb_segmented_image_b[:, :, 2], bins=256)

    plt.plot(bins_red[2:], counts_red[1:], color='r')
    plt.plot(bins_green[2:], counts_green[1:], color='g')
    plt.plot(bins_blue[2:], counts_blue[1:], color='b')
    plt.savefig(f'RGB channels/{image_name}')
    plt.clf()
    hsv_segmented_image = color.rgb2hsv(rgb_segmented_image_b)
    plt.imsave('HSV of Segmented Images/' + curr_image_name + '_channel1.jpg', hsv_segmented_image[:, :, 0])
    plt.imsave('HSV of Segmented Images/' + curr_image_name + '_channel2.jpg', hsv_segmented_image[:, :, 1])
    plt.imsave('HSV of Segmented Images/' + curr_image_name + '_channel3.jpg', hsv_segmented_image[:, :, 2])

    counts_h, bins_h = np.histogram(hsv_segmented_image[:, :, 0], bins=256)
    counts_s, bins_s = np.histogram(hsv_segmented_image[:, :, 1], bins=256)
    counts_v, bins_v = np.histogram(hsv_segmented_image[:, :, 2], bins=256)

    plt.plot(bins_h[10:], counts_h[9:], color='r')
    plt.plot(bins_s[10:], counts_s[9:], color='g')
    plt.plot(bins_v[10:], counts_v[9:], color='b')
    plt.savefig(f'HSV channels/{image_name}')
    plt.clf()

    mask_image[mask_image == 0] = fill_types['implant']
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
    plt.imshow(mask_image_rgb, cmap = 'jet', alpha = 0.35)
    plt.savefig(f'Labelled images/{image_name}')
    plt.clf()
    print('#' + str(n) + ' Binary image saved')
    end = time.time()
    print(f'Time: {end - start}')
    sum_time += end - start
    n += 1

print(f'Average time: {sum_time / (n - 1)}')