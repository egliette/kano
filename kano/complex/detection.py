def extract_bbox_area(image, bbox):
    (left, top), (right, bottom) = bbox
    return image.copy()[top:bottom, left:right]
