from eval import mymetric


#gt = "C:/SR/code/to_subjectify/videos/beach/BasicVSR++_av1_40.mp4"
#dist = "C:/SR/code/to_subjectify/videos/beach/only-codec_x265_2000.mp4"

gt = "C:/SR/GT_YUV/child_1920x1080_24/frame0001.png"
dist = "C:/SR/GT_YUV/child_1920x1080_24/frame0001.png"

print(mymetric(gt, dist, video=False))