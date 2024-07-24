import torch
from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import cv2


def unnormalize_img(img, mean, std):
    """
    img: [3, h, w]
    """
    img = img.detach().cpu().clone()
    # img = img / 255.
    img *= torch.tensor(std).view(3, 1, 1)
    img += torch.tensor(mean).view(3, 1, 1)
    min_v = torch.min(img)
    img = (img - min_v) / (torch.max(img) - min_v)
    return img


def bgr_to_rgb(img):
    return img[:, :, [2, 1, 0]]


def horizon_concate(inp0, inp1):
    h0, w0 = inp0.shape[:2]
    h1, w1 = inp1.shape[:2]
    if inp0.ndim == 3:
        inp = np.zeros((max(h0, h1), w0 + w1, 3), dtype=inp0.dtype)
        inp[:h0, :w0, :] = inp0
        inp[:h1, w0:(w0 + w1), :] = inp1
    else:
        inp = np.zeros((max(h0, h1), w0 + w1), dtype=inp0.dtype)
        inp[:h0, :w0] = inp0
        inp[:h1, w0:(w0 + w1)] = inp1
    return inp


def vertical_concate(inp0, inp1):
    h0, w0 = inp0.shape[:2]
    h1, w1 = inp1.shape[:2]
    if inp0.ndim == 3:
        inp = np.zeros((h0 + h1, max(w0, w1), 3), dtype=inp0.dtype)
        inp[:h0, :w0, :] = inp0
        inp[h0:(h0 + h1), :w1, :] = inp1
    else:
        inp = np.zeros((h0 + h1, max(w0, w1)), dtype=inp0.dtype)
        inp[:h0, :w0] = inp0
        inp[h0:(h0 + h1), :w1] = inp1
    return inp


def transparent_cmap(cmap):
    """Copy colormap and set alpha values"""
    mycmap = cmap
    mycmap._init()
    mycmap._lut[:,-1] = 0.3
    return mycmap

cmap = transparent_cmap(plt.get_cmap('jet'))


def set_grid(ax, h, w, interval=8):
    ax.set_xticks(np.arange(0, w, interval))
    ax.set_yticks(np.arange(0, h, interval))
    ax.grid()
    ax.set_yticklabels([])
    ax.set_xticklabels([])


color_list = np.array(
    [
        0.000, 0.447, 0.741,
        0.850, 0.325, 0.098,
        0.929, 0.694, 0.125,
        0.494, 0.184, 0.556,
        0.466, 0.674, 0.188,
        0.301, 0.745, 0.933,
        0.635, 0.078, 0.184,
        0.300, 0.300, 0.300,
        0.600, 0.600, 0.600,
        1.000, 0.000, 0.000,
        1.000, 0.500, 0.000,
        0.749, 0.749, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 1.000,
        0.667, 0.000, 1.000,
        0.333, 0.333, 0.000,
        0.333, 0.667, 0.000,
        0.333, 1.000, 0.000,
        0.667, 0.333, 0.000,
        0.667, 0.667, 0.000,
        0.667, 1.000, 0.000,
        1.000, 0.333, 0.000,
        1.000, 0.667, 0.000,
        1.000, 1.000, 0.000,
        0.000, 0.333, 0.500,
        0.000, 0.667, 0.500,
        0.000, 1.000, 0.500,
        0.333, 0.000, 0.500,
        0.333, 0.333, 0.500,
        0.333, 0.667, 0.500,
        0.333, 1.000, 0.500,
        0.667, 0.000, 0.500,
        0.667, 0.333, 0.500,
        0.667, 0.667, 0.500,
        0.667, 1.000, 0.500,
        1.000, 0.000, 0.500,
        1.000, 0.333, 0.500,
        1.000, 0.667, 0.500,
        1.000, 1.000, 0.500,
        0.000, 0.333, 1.000,
        0.000, 0.667, 1.000,
        0.000, 1.000, 1.000,
        0.333, 0.000, 1.000,
        0.333, 0.333, 1.000,
        0.333, 0.667, 1.000,
        0.333, 1.000, 1.000,
        0.667, 0.000, 1.000,
        0.667, 0.333, 1.000,
        0.667, 0.667, 1.000,
        0.667, 1.000, 1.000,
        1.000, 0.000, 1.000,
        1.000, 0.333, 1.000,
        1.000, 0.667, 1.000,
        0.167, 0.000, 0.000,
        0.333, 0.000, 0.000,
        0.500, 0.000, 0.000,
        0.667, 0.000, 0.000,
        0.833, 0.000, 0.000,
        1.000, 0.000, 0.000,
        0.000, 0.167, 0.000,
        0.000, 0.333, 0.000,
        0.000, 0.500, 0.000,
        0.000, 0.667, 0.000,
        0.000, 0.833, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 0.167,
        0.000, 0.000, 0.333,
        0.000, 0.000, 0.500,
        0.000, 0.000, 0.667,
        0.000, 0.000, 0.833,
        0.000, 0.000, 1.000,
        0.000, 0.000, 0.000,
        0.143, 0.143, 0.143,
        0.286, 0.286, 0.286,
        0.429, 0.429, 0.429,
        0.571, 0.571, 0.571,
        0.714, 0.714, 0.714,
        0.857, 0.857, 0.857,
        1.000, 1.000, 1.000,
        0.50, 0.5, 0
    ]
).astype(np.float32)
colors = color_list.reshape((-1, 3)) * 255
colors = np.array(colors, dtype=np.uint8).reshape(len(colors), 1, 1, 3)


######################################################################
from lib.config import cfg
import math
def get_img(sampled, coords, bg='w'):
    sampled = sampled.squeeze()
    res = int(cfg.ratio * cfg.H)
    if sampled.dim() == 1:
        sampled = sampled[..., None]
    elif sampled.shape[0] == sampled.shape[1]:
        res = sampled.shape[0]
        if sampled.dim() == 2:
            sampled = sampled[..., None]
        sampled = sampled.reshape(-1, sampled.shape[-1])
    # res = int(math.sqrt(sampled.shape[0]))
    if bg == 'w':
        output = torch.ones(res,res,sampled.shape[-1], dtype=sampled.dtype, device=sampled.device)
    elif bg == 'b':
        output = torch.zeros(res,res,sampled.shape[-1], dtype=sampled.dtype, device=sampled.device)
    else:
        raise NotImplementedError
    x = coords[:,0]
    y = coords[:,1]
    output[x,y] = sampled
    # plt.imshow(output.numpy())
    # plt.show()
    return output

def get_color_from_position(position, image):  # 定义一个函数，输入是一个二维位置和一个图像
    height, width = image.shape[:2]  # 获取图像的高度和宽度

    # 检查输入位置是否在图像边界内，如果不在则返回[-1, -1, -1]
    if position[0] < 0 or position[0] > width - 1 or position[1] < 0 or position[1] > height - 1:
        return np.array([-1., -1., -1.])

    pixel_size = 1.  # 设置像素大小，用于计算插值位置
    # 计算水平方向上的插值位置
    lower_x = np.floor(position[0] / pixel_size) * pixel_size
    upper_x = lower_x + pixel_size
    # 计算垂直方向上的插值位置
    lower_y = np.floor(position[1] / pixel_size) * pixel_size
    upper_y = lower_y + pixel_size

    # 计算四个插值位置的坐标
    lower_left = np.array([lower_x, lower_y])
    lower_right = np.array([upper_x, lower_y])
    upper_left = np.array([lower_x, upper_y])
    upper_right = np.array([upper_x, upper_y])

    # 计算位置相对于插值位置的偏移比例
    horizontal_ratio = (position[0] - lower_x) / pixel_size
    vertical_ratio = (position[1] - lower_y) / pixel_size

    # 对四个插值位置进行处理，计算出每个位置的颜色和面积比例
    # 注意这里使用了np.clip来确保坐标不超过图像的边界
    # 同时，颜色的读取需要注意OpenCV的图像是按照[height, width, channel]的顺序存储的
    lower_left_pix = np.array([np.floor(lower_left[0] + 0.5), np.floor(lower_left[1] + 0.5)])
    lower_left_pix[0] = np.clip(lower_left_pix[0], 0, width - 1)
    lower_left_pix[1] = np.clip(lower_left_pix[1], 0, height - 1)
    lower_left_color = image[int(lower_left_pix[0]), int(lower_left_pix[1])]
    lower_left_area = (1. - horizontal_ratio) * (1. - vertical_ratio)

    lower_right_pix = np.array([np.floor(lower_right[0] + 0.5), np.floor(lower_right[1] + 0.5)])
    lower_right_pix[0] = np.clip(lower_right_pix[0], 0, width - 1)
    lower_right_pix[1] = np.clip(lower_right_pix[1], 0, height - 1)
    lower_right_color = image[int(lower_right_pix[0]), int(lower_right_pix[1])]
    lower_right_area = horizontal_ratio * (1. - vertical_ratio)

    upper_left_pix = np.array([np.floor(upper_left[0] + 0.5), np.floor(upper_left[1] + 0.5)])
    upper_left_pix[0] = np.clip(upper_left_pix[0], 0, width - 1)
    upper_left_pix[1] = np.clip(upper_left_pix[1], 0, height - 1)
    upper_left_color = image[int(upper_left_pix[0]), int(upper_left_pix[1])]
    upper_left_area = (1. - horizontal_ratio) * vertical_ratio

    upper_right_pix = np.array([np.floor(upper_right[0] + 0.5), np.floor(upper_right[1] + 0.5)])
    upper_right_pix[0] = np.clip(upper_right_pix[0], 0, width - 1)
    upper_right_pix[1] = np.clip(upper_right_pix[1], 0, height - 1)
    upper_right_color = image[int(upper_right_pix[0]), int(upper_right_pix[1])]
    upper_right_area = horizontal_ratio * vertical_ratio

    # 根据面积比例计算输入位置的颜色，这就是双线性插值的过程
    interpolated_color = (lower_left_area * lower_left_color + lower_right_area * lower_right_color +
                          upper_left_area * upper_left_color + upper_right_area * upper_right_color) / \
                         (lower_left_area + lower_right_area + upper_left_area + upper_right_area)
    return interpolated_color  # 返回插值后的颜色
def get_color_from_position_torch(position, image):  # 定义一个函数，输入是一个二维位置和一个图像
    height, width = image.shape[:2]  # 获取图像的高度和宽度

    # 检查输入位置是否在图像边界内，如果不在则返回[-1, -1, -1]
    if position[0] < 0 or position[0] > width - 1 or position[1] < 0 or position[1] > height - 1:
        return torch.tensor([-1., -1., -1.])

    pixel_size = 1.  # 设置像素大小，用于计算插值位置
    # 计算水平方向上的插值位置
    lower_x = torch.floor(position[0] / pixel_size) * pixel_size
    upper_x = lower_x + pixel_size
    # 计算垂直方向上的插值位置
    lower_y = torch.floor(position[1] / pixel_size) * pixel_size
    upper_y = lower_y + pixel_size

    # 计算四个插值位置的坐标
    lower_left = torch.tensor([lower_x, lower_y])
    lower_right = torch.tensor([upper_x, lower_y])
    upper_left = torch.tensor([lower_x, upper_y])
    upper_right = torch.tensor([upper_x, upper_y])

    # 计算位置相对于插值位置的偏移比例
    horizontal_ratio = (position[0] - lower_x) / pixel_size
    vertical_ratio = (position[1] - lower_y) / pixel_size

    # 对四个插值位置进行处理，计算出每个位置的颜色和面积比例
    # 注意这里使用了torch.clamp来确保坐标不超过图像的边界
    # 同时，颜色的读取需要注意PyTorch的图像是按照[channel, height, width]的顺序存储的
    lower_left_pix = torch.tensor([torch.floor(lower_left[0] + 0.5), torch.floor(lower_left[1] + 0.5)])
    lower_left_pix[0] = torch.clamp(lower_left_pix[0], 0, width - 1)
    lower_left_pix[1] = torch.clamp(lower_left_pix[1], 0, height - 1)
    lower_left_color = image[int(lower_left_pix[0]), int(lower_left_pix[1])]
    lower_left_area = (1. - horizontal_ratio) * (1. - vertical_ratio)

    lower_right_pix = torch.tensor([torch.floor(lower_right[0] + 0.5), torch.floor(lower_right[1] + 0.5)])
    lower_right_pix[0] = torch.clamp(lower_right_pix[0], 0, width - 1)
    lower_right_pix[1] = torch.clamp(lower_right_pix[1], 0, height - 1)
    lower_right_color = image[int(lower_right_pix[0]), int(lower_right_pix[1])]
    lower_right_area = horizontal_ratio * (1. - vertical_ratio)

    upper_left_pix = torch.tensor([torch.floor(upper_left[0] + 0.5), torch.floor(upper_left[1] + 0.5)])
    upper_left_pix[0] = torch.clamp(upper_left_pix[0], 0, width - 1)
    upper_left_pix[1] = torch.clamp(upper_left_pix[1], 0, height - 1)
    upper_left_color = image[int(upper_left_pix[0]), int(upper_left_pix[1])]
    upper_left_area = (1. - horizontal_ratio) * vertical_ratio

    upper_right_pix = torch.tensor([torch.floor(upper_right[0] + 0.5), torch.floor(upper_right[1] + 0.5)])
    upper_right_pix[0] = torch.clamp(upper_right_pix[0], 0, width - 1)
    upper_right_pix[1] = torch.clamp(upper_right_pix[1], 0, height - 1)
    upper_right_color = image[int(upper_right_pix[0]), int(upper_right_pix[1])]
    upper_right_area = horizontal_ratio * vertical_ratio

    # 根据面积比例计算输入位置的颜色，这就是双线性插值的过程
    interpolated_color = (lower_left_area * lower_left_color + lower_right_area * lower_right_color +
                          upper_left_area * upper_left_color + upper_right_area * upper_right_color) / \
                         (lower_left_area + lower_right_area + upper_left_area + upper_right_area)
    return interpolated_color  # 返回插值后的颜色
def get_colors_from_positions(positions, image):
    # positions should be a tensor of size [N, 2], where N is the number of positions
    height, width = image.shape[:2]
    # invalid_mask = (positions < 0) + (positions > width - 1)
    # invalid_mask = invalid_mask.any(dim=-1)

    pixel_size = 1.
    lower_x = (torch.floor(positions[:, 0] / pixel_size) * pixel_size).unsqueeze(-1)
    upper_x = lower_x + pixel_size
    lower_y = (torch.floor(positions[:, 1] / pixel_size) * pixel_size).unsqueeze(-1)
    upper_y = lower_y + pixel_size

    lower_left = torch.cat([lower_x, lower_y], dim=-1)
    lower_right = torch.cat([upper_x, lower_y], dim=-1)
    upper_left = torch.cat([lower_x, upper_y], dim=-1)
    upper_right = torch.cat([upper_x, upper_y], dim=-1)

    horizontal_ratio = ((positions[:, 0] - lower_x.squeeze()) / pixel_size).unsqueeze(-1)
    vertical_ratio = ((positions[:, 1] - lower_y.squeeze()) / pixel_size).unsqueeze(-1)

    lower_left_pix = torch.floor(lower_left + 0.5)
    # assume width=height
    lower_left_pix = lower_left_pix.clamp(min=0, max=height - 1)
    lower_left_color = image[lower_left_pix[:, 0].long(), lower_left_pix[:, 1].long()]
    lower_left_area = (1. - horizontal_ratio) * (1. - vertical_ratio)

    lower_right_pix = torch.floor(lower_right + 0.5)
    lower_right_pix = lower_right_pix.clamp(min=0, max=height - 1)
    lower_right_color = image[lower_right_pix[:, 0].long(), lower_right_pix[:, 1].long()]
    lower_right_area =  horizontal_ratio * (1. - vertical_ratio)

    upper_left_pix = torch.floor(upper_left + 0.5)
    upper_left_pix = upper_left_pix.clamp(min=0, max=height - 1)
    upper_left_color = image[upper_left_pix[:, 0].long(), upper_left_pix[:, 1].long()]
    upper_left_area = (1. - horizontal_ratio) * vertical_ratio

    upper_right_pix = torch.floor(upper_right + 0.5)
    upper_right_pix = upper_right_pix.clamp(min=0, max=height - 1)
    upper_right_color = image[upper_right_pix[:, 0].long(), upper_right_pix[:, 1].long()]
    upper_right_area = horizontal_ratio * vertical_ratio

    interpolated_colors = (lower_left_area * lower_left_color
                           + lower_right_area * lower_right_color
                           + upper_left_area * upper_left_color
                           + upper_right_area * upper_right_color) / \
                          (lower_left_area
                           + lower_right_area
                           + upper_left_area
                           + upper_right_area)

    # interpolated_colors[invalid_mask] = torch.tensor([-1., -1., -1.])
    return interpolated_colors
# main method
if __name__ == '__main__':
    import torch

    import numpy as np
    import matplotlib.pyplot as plt
    image_path = 'C:/Users/D-Blue/Desktop/cloth/local/surface-aligned-nerf-neus2/data/result/neus_sa_norm_vel\sys-nude_16v_10f_neus_ne-gt_norm_vel-pi-unet/seen_pose_epoch0320_res256_frame0021_vel_emb.jpg'
    # load image
    image = plt.imread(image_path)
    # 图像的大小
    width, height = image.shape

    # 创建一个空的RGB图像
    image = torch.zeros((height, width, 3), dtype=torch.float)

    # 生成渐变效果
    for y in range(height):
        for x in range(width):
            # 从红色渐变到蓝色
            image[y, x] = torch.tensor([x / width , 0, y / width])

    x_coords, y_coords = np.meshgrid(np.arange(128), np.arange(128))
    # 如果你想要一个shape为(height, width, 2)的数组，你可以使用以下代码
    pixel_coordinates = torch.from_numpy(np.stack((y_coords, x_coords), axis=-1)).long()
    target_image = torch.zeros((128, 128, 3), dtype=torch.float)
    target_image2 = torch.zeros((128, 128, 3), dtype=torch.float)
    # print(pixel_coordinates.shape)
    coords = pixel_coordinates.reshape(-1,2)
    print(coords.shape)

    # target_image[coords[:,0],coords[:,1]] = image[coords[:,0]*16//128,coords[:,1]*16//128]
    # for i,coord in enumerate(coords):
    #     if i==127:
    #         print(1)
    #     # color1 = torch.from_numpy(get_color_from_position(coord.numpy()*15/127, image.numpy()))
    #     # target_image[coord[0],coord[1]] = color1
    #
    #     color1 = get_color_from_position_torch(coord*15./127, image)
    #     target_image[coord[0],coord[1]] = color1
    #     color2 = get_colors_from_positions(coords[[i]]*15./127, image)
    #     target_image2[coords[[i],0],coords[[i],1]] = color2
        # if torch.any(color1 != color2[0]):
        #     print(color1,color2)
    target_image2[coords[:,0],coords[:,1]] = get_colors_from_positions(coords*31/127, image)
    # get_colors_from_positions(positions, image)
    # 显示图像

    # show image and target image in one figure
    # plt.subplot(131)
    plt.imshow(image)
    plt.axis('off')
    plt.show()
    # plt.subplot(132)
    # plt.imshow(target_image)
    # plt.subplot(133)
    plt.imshow(target_image2)
    plt.axis('off')
    plt.show()