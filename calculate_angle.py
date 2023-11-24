python
import numpy as np

def calculate_angle(pt1, pt2):
    # 计算两点间的角度
    x_diff = pt2[0] - pt1[0]
    y_diff = pt2[1] - pt1[1]
    return np.degrees(np.arctan2(y_diff, x_diff))

# 在循环中计算车辆行驶角度
for track in tracker.tracks:
    if len(track.locations) > 1:
        angle = calculate_angle(track.locations[-2], track.locations[-1])
        # ...

