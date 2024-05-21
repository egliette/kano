# Image utilities

**Kano** provides some image-related functions:

- `show_image`: show image from a file path or a numpy array.
- `download_image`: download an image from the internet.
- `get_randow_image`: download an image with desired size.
- `rotate_image`: rotate an image around its center.
- `concatenate_images`: concatenate a 2-dimensional list of images.

## `kano.image_utils.show_image`

Show image from a file path or a numpy array.

!!! Example
    ```py
    import cv2
    from kano.image_utils import show_image

    # show image from path
    show_image("your_image.jpg")

    # show image from numpy array
    image = cv2.imread("your_image.jpg")
    show_image(image)
    ```

**Parameters:**

| Name      | Type                  | Description               | Default    |
|-----------|-----------------------|---------------------------|------------|
| `image`   | `str` or `np.ndarray` | Image path or numpy array | _required_ |
| `figsize` | `(float, float)`      | (width, height) for image | None       |
