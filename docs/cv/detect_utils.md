# Object Detection utilites

**Kano** aims to support visualization and extract bounding boxes without the need to handle various box formats, such as xyxy, xywh, or scaled xywh:

- `xywh2xyxy`: convert box with xywh format (x_center, y_center, width, height - which is model input/output format) to xyxy format(x_min, y_min, x_max, y_max - which used to draw boxes).
- `extract_bbox_area`: get cropped box image from the image
- `draw_bbox`: draw bounding box on the image.

::: kano.detect_utils.xywh2xyxy

::: kano.detect_utils.extract_bbox_area

::: kano.detect_utils.draw_bbox
