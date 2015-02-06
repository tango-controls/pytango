class ROI:
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h

    def __repr__(self):
        return str(self)

    def __str__(self):
        txt = "ROI(x={o.x}, y={o.y}, w={o.w}, h={o.h})".format(o=self)
        return txt
