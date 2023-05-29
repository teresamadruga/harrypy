import numpy as np


class BoundingBox:
    """
    A bounding box defined by its top-left and bottom-right coordinates,
    relative to an image with a given width and height. The Bounding Box 
    format is xyxy. The Pascal VOC database uses this format.
    """

    def __init__(
        self, x_top_left: float, y_top_left: float,
        x_bottom_right: float, y_bottom_right: float,
        image_w: int, image_h: int
    ):
        """
        Construct a new BoundingBox instance.

        :param x_top_left: the x-coordinate of the top-left corner.
        :param y_top_left: the y-coordinate of the top-left corner.
        :param x_bottom_right: the x-coordinate of the bottom-right corner.
        :param y_bottom_right: the y-coordinate of the bottom-right corner.
        :param image_w: the width of the image in pixels.
        :param image_h: the height of the image in pixels.
        """
        if not all(
            isinstance(x, (int, 
                           float)) for x in [x_top_left, y_top_left,
                                             x_bottom_right, y_bottom_right,
                                             image_w, image_h]
        ):
            raise TypeError(
                "Bounding box coordinates and image dimensions must be numeric"
            )
        if not (0 <= x_top_left < x_bottom_right <= image_w 
                and 0 <= y_top_left < y_bottom_right <= image_h):
            raise ValueError(
                "Bounding box coordinates must be within image dimensions."
            )
        self._bbox = np.array([x_top_left, y_top_left,
                               x_bottom_right, y_bottom_right],
                              dtype=float)
        self._size = np.array([x_bottom_right - x_top_left,
                               y_bottom_right - y_top_left],
                              dtype=float)
        self._area = np.array([self._size[0] * self._size[1]], dtype=float)
        self._aspect_ratio = np.array([self._size[1] / self._size[0]],
                                      dtype=float)
        self._image_dim = np.array([image_w, image_h], dtype=float)
        self.probs, self._label, self.label_format = None, None, None

    @property
    def bbox(self):
        """
        Return the bounding box coordinates as a numpy array of shape (4,).
        """
        return self._bbox

    @property
    def size(self):
        """
        Return size of the bounding box in width,
        height as a numpy array of shape (2,).
        """
        return self._size

    @property
    def area(self):
        """
        Return the bounding box area in pixels as a numpy array of shape (1,).
        """
        return self._area

    @property
    def aspect_ratio(self):
        """
        Return the aspect ratio as a numpy array of shape (1,).
        """
        return self._aspect_ratio

    @property
    def image_dim(self):
        """
        Return the image dimensions as a numpy array of shape (2,).
        """
        return self._image_dim

    @property
    def bbox_data(self):
        """
        Return a numpy array of shape (6,)
        with the bounding box coordinates
        followed by the image dimensions.
        """
        return np.concatenate((self._bbox, self._image_dim), axis=0)

    def iou(self, box):
        """
        """
        # print(f"{self._bbox = }")
        # print(f"{box = }")
        ix0 = max(self._bbox[0], box[0])
        iy0 = max(self._bbox[1], box[1])
        ix1 = min(self._bbox[2], box[2])
        iy1 = min(self._bbox[3], box[3])

        if ix1 <= ix0 or iy1 <= iy0:
            return 0.0

        intersection_area = (ix1 - ix0) * (iy1 - iy0)
        bbox_area = (box[2] - box[0]) * (box[3] - box[1])
        union_area = self._area + bbox_area - intersection_area

        iou = intersection_area / union_area
        return iou

    def __call__(self):
        """
        Calling the object will return the bounding box coordinates
        as a numpy array of shape (4,).
        """
        return self._bbox

    def xyxy(self):
        """
        Return the bounding box data as a numpy array of shape (4,)
        in xyxy format. The xyxy format uses the top left and bottom right
        coordinates. There's no need of transfomation because
        the BoundingBox class stores data in xyxy format.
        """
        return self._bbox

    def xywh(self):
        """
        Return the bounding box data as a numpy array of shape (4,)
        in xywh format. The xywh format uses the top left coordinate,
        followed by the width and height.
        """
        return np.concatenate((self._bbox[:2], self._size), axis=0)

    def cxcywh(self):
        """
        Return the bounding box data as a numpy array of shape (4,)
        in cxcywh format. The cxcywh format uses the center of the
        box coordinate, followed by the width and height.
        """
        center = self._bbox[:2] + self._size / 2
        # print(f'{center = }')
        return np.concatenate((center, self._size), axis=0)

    def xyxyn(self):
        """
        Return the bounding box data as a numpy array of shape (4,)
        in xyxyn format. The xyxyn format uses the normalized top left
        and bottom right coordinates. The pixel coordinates are divided
        by the image dimensions to be between 0 and 1 => normalized.
        """
        xyxy = self.xyxy()
        x0 = xyxy[0] / self._image_dim[0]
        y0 = xyxy[1] / self._image_dim[1]
        x1 = xyxy[2] / self._image_dim[0]
        y1 = xyxy[3] / self._image_dim[1]
        return np.array([x0, y0, x1, y1], dtype=float)

    def xywhn(self):
        """
        Return the bounding box data as a numpy array of shape (4,)
        in xywhn format. The xywhn format uses the normalized top left
        coordinate, followed by the normalized width and height.
        The pixel coordinates are divided by the image dimensions
        to be between 0 and 1 => normalized.
        """
        xywh = self.xywh()
        x = xywh[0] / self._image_dim[0]
        y = xywh[1] / self._image_dim[1]
        w = xywh[2] / self._image_dim[0]
        h = xywh[3] / self._image_dim[1]
        return np.array([x, y, w, h], dtype=float)

    def cxcywhn(self):
        """
        Return the bounding box data as a numpy array of shape (4,)
        in cxcywhn format. The cxcywhn format uses the normalized
        center of the box coordinate, followed by the normalized
        width and height. The pixel coordinates are divided by
        the image dimensions to be between 0 and 1 => normalized.
        """
        cxcywh = self.cxcywh()
        cx = cxcywh[0] / self._image_dim[0]
        cy = cxcywh[1] / self._image_dim[1]
        w = cxcywh[2] / self._image_dim[0]
        h = cxcywh[3] / self._image_dim[1]
        return np.array([cx, cy, w, h], dtype=float)

    def coco_format(self):
        """
        Return the bounding box data as a numpy array of shape (4,)
        in COCO format. The COCO format or xywh uses the top left coordinate,
        followed by the width and height.
        """
        return self.xywh()

    def pascal_voc_format(self):
        """
        Return the bounding box data as a numpy array of shape (4,)
        in Pascal VOC format. The Pascal VOC format or xyxy uses the top left
        and bottom right coordinates. There's no need of transfomation
        because the BoundingBox class stores data in Pascal VOC - xyxy format.
        """
        return self._bbox

    def yolo_format(self):
        """
        Return the bounding box data as a numpy array of shape (4,)
        in YOLO format. The YOLO format or cxcywhn uses the normalized center
        of the box coordinate, followed by the normalized width and height.
        The pixel coordinates are divided by the image dimensions
        to be between 0 and 1 => normalized.
        """
        return self.cxcywhn()
