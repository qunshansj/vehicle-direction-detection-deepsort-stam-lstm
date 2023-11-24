python
def judge_direction(angle):
    if abs(angle) < 15 or abs(angle - 180) < 15:
        return "直行"
    elif angle > 15 and angle < 165:
        return "左转或左变道"
    elif angle < -15 and angle > -165:
        return "右转或右变道"

# 在循环中判断车辆行驶方向
for track in tracker.tracks:
    if len(track.locations) > 1:
        angle = calculate_angle(track.locations[-2], track.locations[-1])
        direction = judge_direction(angle)
        # ...

