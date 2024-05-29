# Dataset utilities

**Kano** provides some classes to visualize and manipulate YOLO-formatted datasets.

!!! Note
    A valid YOLO-formatted dataset will have a folder tree like this
    ```
        dataset_name:
        ├─ train
        |  ├─ images
        |  |  ├─ image1.jpg
        |  |  └─ ...
        |  └─ labels
        |     ├─ image1.txt
        |     └─ ...
        ├─ valid (optional)
        ├─ test (optional)
        └─ data.yaml (must include a list of classes names inside "names" field)
    ```


- `YoloImage`: visualize, copy a Yolo-formmated image.
- `YoloDataset`: visualize, merge, split Yolo-formatted datasets.


::: kano.dataset_utils.YoloImage

::: kano.dataset_utils.YoloDataset
