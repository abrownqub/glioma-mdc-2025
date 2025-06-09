import config
import cv2
import glioma
import matplotlib.pyplot as plt
import numpy as np


class Transform:

    # super class for all transforms.
    # each transform gets an image to manupulate and a region of interest (roi)
    # which is the area containing the cell
    # the transform manupulates the image and updates the roi accordingly

    # subclasses should implement the transform method which takes an image and returns
    # the transformed image and a dictionary of parameters that describe the
    # transformation (and may be required by the update_roi method)

    # subclasses should also implement the update_roi method which takes the roi and
    # the parameters dictionary and returns the updated roi

    # all transforms except for explicit crop transforms should return an image
    # with the same dimensions as the input image

    def __init__(
        self,
        probability: float = 0.5,  # the probability that the transform will be applied
    ):
        self.probability = probability

    @property
    def should_apply(
        self,
    ) -> bool:
        if np.random.uniform(0, 1) < self.probability:
            return True
        return False

    def transform(
        self,
        cell: glioma.Cell,
    ) -> glioma.Cell:
        return cell

    def __call__(
        self,
        cell: glioma.Cell,
    ) -> glioma.Cell:

        if self.should_apply:
            return self.transform(cell)

        return cell


class TransformSequence(
    Transform,
):

    def __init__(
        self,
        transforms: list[Transform],
    ):
        self.transforms = transforms

    def __call__(
        self,
        cell: glioma.Cell,
        display: bool = False,
    ) -> glioma.Cell:

        transform_count = len(self.transforms)

        if display:
            # setup some plots it's we're tracing the transforms visually
            fig, ax = plt.subplots(
                transform_count + 1, 3, figsize=(10, 5 * (transform_count + 1))
            )

            def plot(cell, index, title):
                cell.image.plot_padded(ax[index, 0])
                cell.plot_polygon(ax[index, 0])
                cell.plot_bounding_box(ax[index, 0])
                cell.plot_patch_box(ax[index, 0])
                cell.plot_W(ax[index, 1])
                cell.plot_X(ax[index, 2])
                ax[index, 0].set_title(title)

            plot(cell, 0, "original")

        ii = 1
        for transform in self.transforms:

            cell = transform(cell)

            if display:
                plot(cell, ii, transform.__class__.__name__)
                ii += 1

        return cell


class RandomRotate(
    Transform,
):

    # performs a rotation by a random angle

    def __init__(
        self,
        probability: float,
    ):
        super().__init__(probability)

    def transform(
        self,
        cell: glioma.Cell,
    ) -> glioma.Cell:

        angle = np.random.uniform(0, 360)
        angle = np.radians(angle)

        center = cell.X.shape[1] // 2, cell.X.shape[0] // 2

        patch_rotation_matrix = cv2.getRotationMatrix2D(
            angle=np.degrees(angle),
            center=center,
            scale=1,
        )

        cell.X = cv2.warpAffine(
            cell.X,
            patch_rotation_matrix,
            (cell.X.shape[1], cell.X.shape[0]),
        )

        box_rotation_matrix = np.array(
            [
                [np.cos(-angle), -np.sin(-angle)],
                [np.sin(-angle), np.cos(-angle)],
            ]
        )

        cell.patch_box_offsets = np.dot(cell.patch_box_offsets, box_rotation_matrix)

        return cell


class RandomHFlip(
    Transform,
):

    def transform(
        self,
        cell: glioma.Cell,
    ) -> glioma.Cell:

        cell.X = cv2.flip(cell.X, 1)

        cell.patch_box_offsets = np.array(
            [
                cell.patch_box_offsets[1, :],  # top_right
                cell.patch_box_offsets[0, :],  # top_left
                cell.patch_box_offsets[3, :],  # bottom_left
                cell.patch_box_offsets[2, :],  # bottom_right
            ]
        )

        return cell


class RandomVFlip(
    Transform,
):

    # performs a random vertical flip

    def transform(
        self,
        cell: glioma.Cell,
    ) -> glioma.Cell:

        cell.X = cv2.flip(cell.X, 0)

        cell.patch_box_offsets = np.array(
            [
                cell.patch_box_offsets[3, :],  # bottom_left
                cell.patch_box_offsets[2, :],  # bottom_right
                cell.patch_box_offsets[1, :],  # top_right
                cell.patch_box_offsets[0, :],  # top_left
            ]
        )

        return cell


class Turn(
    Transform,
):

    # performs a 90 clockwise degree rotation

    def should_apply(
        self,
    ) -> bool:
        return True

    def transform(
        self,
        cell: glioma.Cell,
    ) -> glioma.Cell:

        cell.X = cv2.rotate(cell.X, cv2.ROTATE_90_CLOCKWISE)

        original_top = cell.patch_box_offsets[0, 1]
        original_left = cell.patch_box_offsets[0, 0]
        original_bottom = cell.patch_box_offsets[2, 1]
        original_right = cell.patch_box_offsets[2, 0]

        turned_top = original_left
        turned_right = original_top
        turned_bottom = original_right
        turned_left = original_bottom

        turned_top_left = np.array([turned_top, turned_left])
        turned_top_right = np.array([turned_top, turned_right])
        turned_bottom_left = np.array([turned_bottom, turned_left])
        turned_bottom_right = np.array([turned_bottom, turned_right])

        cell.patch_box_offsets = np.array(
            [
                turned_top_left,
                turned_top_right,
                turned_bottom_right,
                turned_bottom_left,
            ]
        )

        return cell


class RandomXScale(
    Transform,
):
    # takes a (H, W, C) image and effectively grows or shrinks the image in the
    # x-direction by cropping/padding the image and then resizing so that the
    # the output image is the same size as the original image but the content is
    # scaled in the x direction
    #
    # you set the probability this happens
    # you set the maximum amount of cropping/padding to limit the scaling
    #
    # this means you can ensure the size of the content for later cropping

    def __init__(
        self,
        probability: float,
        max_crop: int,
        max_pad: int,
    ):
        super().__init__(probability)
        self.max_crop = max_crop
        self.max_pad = max_pad

    def shrink(
        self,
        cell: glioma.Cell,
        pad: int = None,
    ) -> glioma.Cell:

        if pad is None:
            pad = np.random.randint(1, self.max_pad)
            pad = self.max_pad

        left_pad = pad // 2
        right_pad = pad - left_pad

        original_shape = cell.X.shape

        cell.X = cv2.copyMakeBorder(
            src=cell.X,
            top=0,
            bottom=0,
            left=left_pad,
            right=right_pad,
            borderType=cv2.BORDER_CONSTANT,
            value=0,
        )

        cell.patch_box_offsets[:, 0] = (
            (cell.patch_box_offsets[:, 0] * cell.X.shape[1]) / original_shape[1]
        ).astype(int)

        cell.X = cv2.resize(cell.X, original_shape[:2])

        return cell

    def grow(
        self,
        cell: glioma.Cell,
        crop: int = None,
    ) -> glioma.Cell:

        if crop is None:
            crop = np.random.randint(1, self.max_crop)
            crop = self.max_crop

        left_crop = crop // 2
        right_crop = crop - left_crop

        original_shape = cell.X.shape

        cell.X = cell.X[:, left_crop:-right_crop]

        cell.patch_box_offsets[:, 0] = (
            (cell.patch_box_offsets[:, 0] * cell.X.shape[1]) / original_shape[1]
        ).astype(int)

        cell.X = cv2.resize(cell.X, original_shape[:2])

        return cell

    def transform(
        self,
        cell: glioma.Cell,
    ) -> glioma.Cell:

        should_skrink = np.random.uniform(0, 1) < 0.5

        if should_skrink:
            return self.shrink(cell)

        return self.grow(cell)


class RandomXShift(
    Transform,
):
    # takes a (H, W, C) image and shifts the image in the x direction
    # by cropping/padding the image on the left/right.  the output image is the
    # same size as the original image but the content is shifted in the x direction
    #
    # you set the probability this happens
    # you set the maximum amount of the shift
    #
    # this means you can ensure the size of the content for later cropping

    def __init__(
        self,
        probability: float,
        max_left_shift: int,
        max_right_shift: int,
    ):
        super().__init__(probability)
        self.max_left_shift = max_left_shift
        self.max_right_shift = max_right_shift

    def transform(
        self,
        cell: glioma.Cell,
    ) -> glioma.Cell:

        shift = np.random.randint(-self.max_left_shift, self.max_right_shift)

        # a positive shift means move the image right, so

        left_pad = max(0, shift)
        right_pad = max(0, -shift)

        cell.X = cv2.copyMakeBorder(
            src=cell.X,
            top=0,
            bottom=0,
            left=left_pad,
            right=right_pad,
            borderType=cv2.BORDER_CONSTANT,
            value=0,
        )

        # a positive shift means we need to crop the right

        left_crop = max(0, -shift)
        right_crop = max(0, shift)

        if right_crop == 0:
            cell.X = cell.X[:, left_crop:]
        if right_crop > 0:
            cell.X = cell.X[:, left_crop:-right_crop]

        cell.patch_box_offsets[:, 0] += shift

        return cell


class CenterBoxCrop(
    Transform,
):

    # this takes a (TARGET_SIZE, TARGET_SIZE, C) crop of the image from it's centre

    def should_apply(
        self,
    ) -> bool:
        return True

    def transform(
        self,
        cell: glioma.Cell,
    ) -> glioma.Cell:

        shape = cell.X.shape

        center_x = shape[1] // 2
        center_y = shape[0] // 2
        crop_size = config.NETWORK_INPUT_SIZE  # / cell.scale

        top = int(center_y - crop_size / 2)
        bottom = top + crop_size
        left = int(center_x - crop_size / 2)
        right = left + crop_size

        crop = cell.X[
            top:bottom,
            left:right,
        ]

        # cell.X = cv2.resize(
        #     crop, (config.NETWORK_INPUT_SIZE, config.NETWORK_INPUT_SIZE)
        # )

        cell.X = crop

        return cell


class RGBLevels(
    Transform,
):

    def __init__(
        self,
        probability: float,
        red_range: tuple[int, int] = (-50, 50),
        green_range: tuple[int, int] = (-30, 30),
        blue_range: tuple[int, int] = (-80, 80),
    ):
        super().__init__(probability)
        self.red_range = red_range
        self.green_range = green_range
        self.blue_range = blue_range

    def transform(
        self,
        cell: glioma.Cell,
    ) -> glioma.Cell:

        b, g, r = cv2.split(cell.X)
        r = cv2.add(r, np.random.randint(*self.red_range))
        g = cv2.add(g, np.random.randint(*self.green_range))
        b = cv2.add(b, np.random.randint(*self.blue_range))
        cell.X = cv2.merge((b, g, r))

        return cell


class Saturation(
    Transform,
):

    def __init__(
        self,
        probability: float,
        saturation_range: tuple[int, int] = (-100, 100),
    ):
        super().__init__(probability)
        self.saturation_range = saturation_range

    def transform(
        self,
        cell: glioma.Cell,
    ) -> glioma.Cell:

        x = cell.X
        x = cv2.cvtColor(x, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(x)
        s = cv2.add(s, np.random.randint(*self.saturation_range))
        x = cv2.merge((h, s, v))
        x = cv2.cvtColor(x, cv2.COLOR_HSV2BGR)
        cell.X = x

        return cell

    # def transform(
    #     self,
    #     cell: glioma.Cell,
    # ) -> glioma.Cell:

    #     x = cv2.cvtColor(x, cv2.COLOR_BGR2HSV)
    #     h, s, v = cv2.split(x)
    #     s = cv2.add(s, np.random.randint(*self.saturation_range))
    #     x = cv2.merge((h, s, v))
    #     x = cv2.cvtColor(x, cv2.COLOR_HSV2BGR)

    #     return x, {}


class Brightness(
    Transform,
):

    def __init__(
        self,
        probability: float,
        brightness_range: tuple[int, int] = (-100, 100),
    ):
        super().__init__(probability)
        self.brightness_range = brightness_range

    def transform(
        self,
        cell: glioma.Cell,
    ) -> glioma.Cell:

        x = cell.X
        x = cv2.cvtColor(x, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(x)
        v = cv2.add(v, np.random.randint(*self.brightness_range))
        x = cv2.merge((h, s, v))
        x = cv2.cvtColor(x, cv2.COLOR_HSV2BGR)
        cell.X = x

        return cell

    # def transform(
    #     self,
    #     cell: glioma.Cell,
    # ) -> glioma.Cell:

    #     x = cv2.cvtColor(x, cv2.COLOR_BGR2HSV)
    #     h, s, v = cv2.split(x)
    #     v = cv2.add(v, np.random.randint(*self.brightness_range))
    #     x = cv2.merge((h, s, v))
    #     x = cv2.cvtColor(x, cv2.COLOR_HSV2BGR)

    #     return x, {}


class GammaContrast(
    Transform,
):

    def __init__(
        self,
        probability: float,
        gamma_range: tuple[float, float] = (0.5, 2.0),
    ):
        super().__init__(probability)
        self.gamma_range = gamma_range

    def transform(
        self,
        cell: glioma.Cell,
    ) -> glioma.Cell:

        gamma = np.random.uniform(*self.gamma_range)
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)])

        x = cell.X
        x = cv2.LUT(x, table.astype(np.uint8))
        cell.X = x

        return cell

    # def transform(
    #     self,
    #     cell: glioma.Cell,
    # ) -> glioma.Cell:

    #     gamma = np.random.uniform(*self.gamma_range)

    #     inv_gamma = 1.0 / gamma
    #     table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)])
    #     x = cv2.LUT(x, table.astype(np.uint8))

    #     return x, {}


class CLAHE(
    Transform,
):

    def __init__(
        self,
        probability: float,
        clip_range: tuple[int, int] = (1, 3),
        roi: tuple[int, int, int, int] = None,
    ):
        super().__init__(probability)
        self.clip_range = clip_range

    def transform(
        self,
        cell: glioma.Cell,
    ) -> glioma.Cell:

        clahe = cv2.createCLAHE(np.random.uniform(*self.clip_range))

        x = cell.X
        b, g, r = cv2.split(x)
        r = clahe.apply(r)
        g = clahe.apply(g)
        b = clahe.apply(b)
        x = cv2.merge((b, g, r))
        cell.X = x

        return cell

    # def transform(
    #     self,
    #     cell: glioma.Cell,
    # ) -> glioma.Cell:

    #     clahe = cv2.createCLAHE(np.random.uniform(*self.clip_range))

    #     b, g, r = cv2.split(x)
    #     r = clahe.apply(r)
    #     g = clahe.apply(g)
    #     b = clahe.apply(b)
    #     x = cv2.merge((b, g, r))

    #     return x, {}


class Equalize(
    Transform,
):

    def __init__(
        self,
        probability: float,
    ):
        super().__init__(probability)

    def transform(
        self,
        cell: glioma.Cell,
    ) -> glioma.Cell:

        x = cell.X
        b, g, r = cv2.split(x)
        r = cv2.equalizeHist(r)
        g = cv2.equalizeHist(g)
        b = cv2.equalizeHist(b)
        x = cv2.merge((b, g, r))
        cell.X = x

        return cell

    # def transform(
    #     self,
    #     cell: glioma.Cell,
    # ) -> glioma.Cell:

    #     b, g, r = cv2.split(x)
    #     r = cv2.equalizeHist(r)
    #     g = cv2.equalizeHist(g)
    #     b = cv2.equalizeHist(b)
    #     x = cv2.merge((b, g, r))

    #     return x, {}


class ResetCell(
    Transform,
):

    def should_apply(
        self,
    ) -> bool:
        return True

    def transform(
        self,
        cell: glioma.Cell,
    ) -> glioma.Cell:

        # cell.reset()
        cell.reset()
        cell._X = None
        return cell


class HSD(
    Transform,
):

    def __init__(
        self,
        probability=0.5,
        variance=0.2,
        epsilon=1e-8,
    ):
        super().__init__(probability)
        self.epsilon = epsilon
        self.variance = variance

    def transform(
        self,
        cell: glioma.Cell,
    ) -> glioma.Cell:

        x = cell.X  # (H, W, C) [0, 255]

        x = x / 255.0  # (H, W, C) [0, 1]

        # to_od

        x = x.clip(self.epsilon, 1)  # (H, W, C) [epsilon, 1]
        x = -np.log(x)  # (H, W, C) [0, inf]

        # od_to_hsd

        x = x.clip(self.epsilon)  # (H, W, C) [epsilon, inf]

        D = x.mean(axis=(-1), keepdims=False)  # (H, W) [epsilon, inf]

        cx = x[:, :, 2] / D - 1  # RED
        cy = (x[:, :, 1] - x[:, :, 0]) / (D * np.sqrt(3.0))  # GREEN - BLUE

        # shift_cx_cy

        cx = cx + np.random.normal(0, self.variance, (1,))
        cy = cy + np.random.normal(0, self.variance, (1,))

        # hsd_to_od

        I_red = D * (cx + 1)
        I_green = 0.5 * D * (2 - cx + (np.sqrt(3.0) * cy))
        I_blue = 0.5 * D * (2 - cx - (np.sqrt(3.0) * cy))

        od = np.stack([I_blue, I_green, I_red], axis=-1)

        # from_od

        od = od.clip(self.epsilon)

        od = np.exp(-od)

        od = od * 255

        od = od.astype(np.uint8)

        cell.X = od

        return cell


class Blur(Transform):

    def __init__(
        self,
        probability=0.5,
        kernel_size=5,
        fade=1.0,
        mask_kernel_size=25,
    ):
        super().__init__(probability)
        self.kernel_size = kernel_size
        self.fade = fade
        self.mask_kernel_size = mask_kernel_size

    def transform(
        self,
        cell: glioma.Cell,
    ) -> glioma.Cell:

        patch = cell.W.copy()

        x = cell.X

        # create poygon mask
        mask = np.zeros_like(patch[:, :, 0], dtype=np.float32)
        offsets = cell.polygon_offsets

        # subtract from all negative values and add to all positive values
        step = 10
        offsets = np.where(offsets < 0, offsets - step, offsets)
        offsets = np.where(offsets > 0, offsets + step, offsets)
        absolute_polygon = x.shape[0] / 2.0 + offsets
        offsets = np.where(offsets < 0, 0, offsets)
        offsets = np.where(offsets >= x.shape[0], x.shape[0] - 1, offsets)

        # TODO rounding
        absolute_polygon = absolute_polygon.round().astype(np.int32)
        cv2.fillPoly(mask, [absolute_polygon], 1.0)
        mask = cv2.GaussianBlur(mask, (self.mask_kernel_size, self.mask_kernel_size), 0)
        mask = np.stack([mask] * 3, axis=-1)

        # get polygon mask & set other pixels to black
        # extracted_patch = np.zeros_like(patch)
        # for c in range(3):  # Assuming RGB image
        #     extracted_patch[:, :, c] = cv2.bitwise_and(patch[:, :, c], mask)

        # blur

        kernel = int(self.kernel_size) * 2 + 1
        blurred_patch = cv2.GaussianBlur(x, (kernel, kernel), 0)
        blurred_patch = blurred_patch * self.fade
        blurred_patch = blurred_patch.astype(np.uint8)

        result_patch = x * (mask) + blurred_patch * (1.0 - mask)
        result_patch = result_patch.astype(np.uint8)

        cell.X = result_patch

        return cell


class PolygonOnly(Transform):

    def transform(
        self,
        cell: glioma.Cell,
    ) -> glioma.Cell:

        patch = cell.W.copy()

        x = cell.X

        # create polygon mask
        mask = np.zeros_like(patch[:, :, 0], dtype=np.uint8)
        absolute_polygon = x.shape[0] / 2 + cell.polygon_offsets
        cv2.fillPoly(mask, [absolute_polygon.astype(np.int32)], 255)

        # get polygon mask & set other pixels to black
        extracted_patch = np.zeros_like(patch)
        for c in range(3):
            extracted_patch[:, :, c] = cv2.bitwise_and(patch[:, :, c], mask)

        cell.X = extracted_patch

        return cell
