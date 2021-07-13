from os import listdir
from os.path import isfile, join
import time
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
            if above_avg > (mask_size * mask_size) * 0.6:
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


image_names = [f for f in listdir('Implantatum-KA_images') if isfile(join('Implantatum-KA_images', f))]

n = 0
for name in image_names:
    print(f'{n}: {name}')
    n += 1

n = 1
sum_time = 0

for i in range(37,38):
    image_name = image_names[i]
    type = ''
    print(f'Processing {str(n)} of {str(len(image_names))} images. Image name: {image_name}')
    curr_image_name = image_name
    #image = img.imread('2020-11-02/115_2418_25_7_5x.jpg')
    start = time.time()
    rgb_image = img.imread('Implantatum-KA_images/' + image_name)
    hsv_image = color.rgb2hsv(rgb_image)
    # fig, ax = plt.subplots(1,3)
    # ax[0].imshow(hsv_image[:, :, 0])
    # ax[1].imshow(hsv_image[:, :, 1])
    # ax[2].imshow(hsv_image[:, :, 2])
    # plt.show()
    threshold = filters.threshold_mean(hsv_image[:,:,0])
    gray_image = hsv_image[:,:,2]
    gray_image[hsv_image[:,:,0] > threshold] = 0
    # plt.imshow(gray_image)
    # plt.show()
    # plt.imshow((gray_image_2 + gray_image) / 2)
    # plt.show()

    image = (color.rgb2gray(rgb_image) * 255).astype(np.uint8)
    #edge_image = feature.canny(edge_image, sigma=1)
    edge_image = filters.sobel(gray_image)
    #edge_image = morphology.closing(edge_image, selem=morphology.square(5))
    # plt.imshow(edge_image)
    # plt.show()
    plt.imsave('Edge images/' + image_name, edge_image, cmap = 'gray')
    #edge_image = get_binary_image(edge_image)
    edge_image = get_binary_image_from_sobel(edge_image, mask_size=15, step_size=15)
    edge_image, boundary_coords = get_contour(edge_image)
    # boundary_coords[0] = boundary_coords[0] + 350
    boundary_coords[1] = boundary_coords[1] - 300
    # boundary_coords[2] = boundary_coords[2] + 100
    #boundary_coords[3] = boundary_coords[3] - 350
    segmented_image = image
    #segmented_image[edge_image == 0] = 0
    segmented_image = segmented_image[boundary_coords[0]: boundary_coords[1], boundary_coords[2]: boundary_coords[3]]
    mask_size = 196
    bin_segmented_image = np.zeros_like(segmented_image)
            #bin_segmented_image[i:i + mask_size, j: j + mask_size] = temp_image
            # plt.imshow(cluster_image)
            # plt.show()
    # plt.imshow(bin_segmented_image)
    # plt.show()
    #segmented_image[edge_image == 0] = 0
    rgb_segmented_image = np.zeros_like(rgb_image)
    coords = np.where(edge_image != 0)
    for i in range(len(coords[0])):
        rgb_segmented_image[coords[0][i], coords[1][i]] = rgb_image[coords[0][i], coords[1][i]]
    rgb_segmented_image = rgb_segmented_image[boundary_coords[0]: boundary_coords[1], boundary_coords[2]: boundary_coords[3]]
    # plt.imshow(rgb_segmented_image)
    # plt.show()
    # rgb_segmented_image[630:,:1075] = [0,0,0]
    # rgb_segmented_image[850:,1600:] = [0,0,0]
    edge_image = edge_image[boundary_coords[0]: boundary_coords[1], boundary_coords[2]: boundary_coords[3]]
    # plt.imshow(rgb_segmented_image)
    # plt.show()
    # patch_n = 0
    # for i in range(0, rgb_segmented_image.shape[0], mask_size):
    #     for j in range(0, rgb_segmented_image.shape[1], mask_size):
    #         temp_image = rgb_segmented_image[i:i + mask_size, j: j + mask_size]
    #         if temp_image.shape[0] < mask_size or temp_image.shape[1] < mask_size:
    #             temp_zeros = np.zeros((mask_size, mask_size, 3))
    #             temp_zeros[:temp_image.shape[0], :temp_image.shape[1]] = temp_image
    #             temp_image = temp_zeros.astype(np.uint8)
    #         print(temp_image.shape)
    #         hsv_image = color.rgb2hsv(temp_image)
    #         # fig, ax = plt.subplots(1,3)
    #         # ax[0].imshow(hsv_image[:, :, 0])
    #         # ax[1].imshow(hsv_image[:, :, 1])
    #         # ax[2].imshow(hsv_image[:, :, 2])
    #         # plt.show()
    #         map_image = hsv_image[:, :, 2]
    #         map_image = smooth_map_image(map_image)
    #         patch_n += 1
    #         mask_image = np.zeros((temp_image.shape[0], temp_image.shape[1]))
    #         current_coords = [100,60]
    #         temp_edge_image = color.rgb2gray(filters.sobel(map_image))
    #         mask_image = snake_fill(mask_image, temp_edge_image, map_image, current_coords)
    #         fig, ax = plt.subplots(1,3)
    #         ax[0].imshow(map_image)
    #         ax[1].imshow(temp_edge_image)
    #         ax[2].imshow(mask_image)
    #         plt.show()
            # try:
            #     plt.imsave(f'Image patches/{image_name}/{image_name}patch_{i}-{j}.png', temp_image)
            # except:
            #     os.mkdir('Image patches/' + image_name)
            #     plt.imsave(f'Image patches/{image_name}/{image_name}patch_{i}-{j}.png', temp_image)
    # rgb_segmented_image = rgb_segmented_image[boundary_coords[0]: boundary_coords[1], boundary_coords[2]: boundary_coords[3]]

    # plt.imshow(segmented_image)
    # plt.show()
    #get_implanted_area(rgb_segmented_image)
    # plt.show()
    #plt.imshow(edge_image)
    #image[edge_image == 0] //= 3
    plt.imsave('Segmented images/' + image_name, rgb_segmented_image)
    plt.imsave('Binary images/' + image_name, edge_image, cmap = 'gray')
    plt.clf()
    print('#' + str(n) + ' Binary image saved')
    end = time.time()
    print(f'Time: {end - start}')
    sum_time += end - start
    n += 1