import os
import json

from eval import mymetric


'''
gt0 = "C:/SR/code/to_subjectify/videos/beach/BasicVSR++_av1_40.mp4"
dist0 = "C:/SR/code/to_subjectify/videos/beach/only-codec_x265_2000.mp4"

gt = "C:/Users/evgen/Videos/bs_Folder/bs1_25fps.yuv"
dist = "C:/Users/evgen/Videos/bs_Folder/bs3_25fps.yuv"

gt_par = {
    "video_format" : "YUV420",
    "video_size" : (768, 432),
    "fps" : 25
}

dist_par = gt_par

#print("1___", mymetric(gt, dist, video=True, gt_params=gt_par, dist_params=dist_par), "___")

#print("2___", mymetric(gt0, dist0, video=True), "___")

#print("3___", mymetric(gt0, dist, video=True, dist_params=dist_par), "___")
'''

gt_par = {
    "video_format" : "YUV420",
    "video_size" : (768, 432),
    "fps" : 25
}


with open("../live_subj_data.json", "r") as f:
    data = json.load(f)


def store(key, value):
    save = "./result2.json"
    if not os.path.exists(save):
        data = {}
    else:
        with open(save, "r") as f:
            data = json.load(f)
    data[key] = value
    with open(save, "w") as f:
        json.dump(data, f)

gts = "C:/Users/evgen/Videos/down/videos"
dists = "C:/Users/evgen/Videos/down/compressed_videos"

for folder in os.listdir(gts):
    for video in os.listdir(gts + "/" + folder):
        number = video[2]
        if number == "1" and video[3] == "_":
            continue

        fps_part = list(video.split("_"))[1]
        fps = int(fps_part[:-7])

        gt_par["fps"] = fps
        
        #dist = dists + "/" + folder + "/" + video
        #print(dist)
        
        
        with open("./result2.json", "r") as f:
            tmp = json.load(f)
        if video in tmp.keys():
            continue
        
        
        
        #name = list(folder.split("_"))[0]
        gt = gts + "/" + folder + "/" + video[:2] + "1" + video[3:]
        dist = gts + "/" + folder + "/" + video
        print(gt, dist)
        subj_value = data[video[:-4] + ".yuv"]

        print(subj_value)
        metr_value = mymetric(gt, dist, True, gt_par, gt_par)
        print(metr_value)
        store(video, [subj_value, metr_value])
        '''
        if dist.endswith(".m2v"):
            continue

        if dist.endswith(".mp4") or dist.endswith(".264"):
            dist_par = None
        else:
            dist_par = gt_par

        print(subj_value)
        metr_value = mymetric(gt, dist, True, gt_par, dist_par)
        print(metr_value)

        if metr_value != -1:
            store(video, [subj_value, metr_value])
        else:
            metr_value = mymetric(gt, dist, True, gt_par, gt_par)
            if metr_value != -1:
                store(video, [subj_value, metr_value])
        '''
