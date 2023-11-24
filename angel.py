python


class AngleCalculator:
    def __init__(self, p1, p2):
        self.p1 = p1
        self.p2 = p2

    def angle_between(self):
        ang1 = np.arctan2(*self.p1[::-1])
        ang2 = np.arctan2(*self.p2[::-1])
        return np.rad2deg((ang1 - ang2) % (2 * np.pi))

# 315.
