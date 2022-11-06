import os
import cv2
import numpy as np
import lpips
import torch
from skimage import color
from erqa import ERQA


def calculate_ERQA_and_LPIPS_and_color_on_video(gt_path, dist_path, gt_params=None, dist_params=None):
    metric = ERQA()
    loss_fn_alex = lpips.LPIPS(net='alex')
    
    gt_path = check_file_format(gt_path, gt_params, gt=True)
    dist_path = check_file_format(dist_path, dist_params, gt=False)

    vidcap_gt = cv2.VideoCapture(gt_path)
    success_gt, image_gt = vidcap_gt.read()

    vidcap_dist = cv2.VideoCapture(dist_path)
    success_dist, image_dist = vidcap_dist.read()
    
    values_erqa = []
    values_lpips = []
    values_color = []
    counter = 0
    while success_gt and success_dist:
        if counter % 5 == 0:        
            # ERQA
            value = metric(image_dist, image_gt)
            values_erqa.append(value)

            # Colorfulness
            colorfulness = calculate_colorfullness(image_dist)
            values_color.append(colorfulness)

            # LPIPS
            image_gt = image_gt.transpose((2, 0, 1))
            image_dist = image_dist.transpose((2, 0, 1))

            gt = torch.tensor(image_gt.reshape(1, *image_gt.shape))
            dist = torch.tensor(image_dist.reshape(1, *image_gt.shape))

            value = loss_fn_alex(gt, dist)
            values_lpips.append(value.detach()[0][0][0][0])
        counter += 1

        success_gt, image_gt = vidcap_gt.read()
        success_dist, image_dist = vidcap_dist.read()
        
    return np.mean(values_erqa), np.mean(values_lpips), np.mean(values_color)


def calculate_ERQA_and_LPIPS_and_color_on_frame(gt_path, dist_path):
    metric = ERQA()
    loss_fn_alex = lpips.LPIPS(net='alex')

    image_gt = cv2.imread(gt_path)
    image_dist = cv2.imread(dist_path)

    value_erqa = metric(image_dist, image_gt)
    colorfulness = calculate_colorfullness(image_dist)

    image_gt = image_gt.transpose((2, 0, 1))
    image_dist = image_dist.transpose((2, 0, 1))

    gt = torch.tensor(image_gt.reshape(1, *image_gt.shape))
    dist = torch.tensor(image_dist.reshape(1, *image_gt.shape))

    value_lpips = loss_fn_alex(gt, dist)
    value_lpips = value_lpips.detach()[0][0][0][0]

    return value_erqa, value_lpips, colorfulness


def calculate_MDTVSFA(video_path, video_params=None):
    if video_params is None:
        os.system(f"python MDTVSFA/test_demo.py --model_path=MDTVSFA/models/MDTVSFA.pt \
            --video_path={video_path} > res.txt")
    else:
        video_size = video_params["video_size"]
        os.system(f"python MDTVSFA/test_demo.py --model_path=MDTVSFA/models/MDTVSFA.pt \
            --video_path={video_path} --video_width {video_size[0]} --video_height {video_size[1]}\
                 --video_format {video_params['video_format']}> res.txt")
    with open("res.txt", "r") as f:
        lines = list(f.readlines())
        for i, line in enumerate(lines):
            if "Predicted perceptual quality:" in line:
                words = list(line.split("["))
                value = words[1][:-2]
                break
    if os.path.exists("res.txt"):
        os.remove("res.txt")

    try:
        return float(value)
    except:
        return -1


def calculate_SI_TI(video_path, video_params=None):
    add = ""
    if video_params is not None:
        if video_params["video_format"] == "YUV420":
            add = f" -s {video_params['video_size'][0]}x{video_params['video_size'][1]} \
                -r {video_params['fps']} -pix_fmt yuv420p "
    
    path_to_encoded = "tmp.mp4"
    cmd = f'ffmpeg -hide_banner -loglevel error {add} -i {video_path} -c:v libx264 -qp 28 -b_qfactor 1 -i_qfactor 1 -an {path_to_encoded}'
    os.system(cmd)
    
    cmd = f'ffprobe -hide_banner -loglevel error -show_frames {path_to_encoded} > tmp_output.txt'
    os.system(cmd)

    with open('tmp_output.txt', 'r') as f:
        result = f.readlines()

        frame_width = None
        frame_height = None

        for line in result:
            if frame_width and frame_height:
                break
            if 'width' in line:
                frame_width = int(line.split("=")[1])
            if 'height' in line:
                frame_height = int(line.split("=")[1])

        result = list(
            filter(
                lambda x: ("pkt_size" in x)
                or ("pict_type" in x)
                or ("coded_picture_number" in x),
                result,
            )
        )

        result = [
            {
                size.split("=")[0]: int(size.split("=")[1]),
                type_frame.split("=")[0]: type_frame.split("=")[1].strip(),
                number.split("=")[0]: int(number.split("=")[1]),
            }
            for size, type_frame, number in zip(result[::3], result[1::3], result[2::3])
        ]

        I_sum = np.mean(
            [frame["pkt_size"] for frame in result if frame["pict_type"] == "I"]
        )
        P_sum = np.mean(
            [frame["pkt_size"] for frame in result if frame["pict_type"] == "P"]
        )
        SI = I_sum / (3 * frame_width * frame_height / 2)
        TI = P_sum / I_sum

    if os.path.exists(path_to_encoded):
        os.remove(path_to_encoded)
    if os.path.exists("tmp_output.txt"):
        os.remove("tmp_output.txt")

        return SI, TI


def calculate_colorfullness(image):
    lab = color.rgb2lab(image)
    hsv = color.rgb2hsv(image)

    chroma = hsv[..., 1]

    L = lab[..., 0]
    a = lab[..., 1]
    b = lab[..., 2]

    sigma_a = a.std()
    sigma_b = b.std()

    sigma = np.sqrt(sigma_a ** 2 + sigma_b ** 2)
    mean_chroma = chroma.mean()

    return sigma + 0.94 * mean_chroma


def check_file_format(video_path, video_params, gt=False):
    if video_params is None:
        return video_path
    if video_params["video_format"] == "YUV420":
        if gt:
            tmp_path = "./__gt.mp4"
        else:
            tmp_path = "./__dist.mp4"
        os.system(f"ffmpeg -y -hide_banner -loglevel error -s {video_params['video_size'][0]}x{video_params['video_size'][1]} -r {video_params['fps']}\
            -pix_fmt yuv420p -i {video_path} -c:v libx264 -crf 0 {tmp_path}")
        return tmp_path
    assert video_params["video_format"] == 'YUV420' or video_params["video_format"] == 'RGB'
    return video_path

        