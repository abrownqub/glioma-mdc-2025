import base64
import json
import os

import config
import cv2
import data

# import data
import matplotlib.pyplot as plt
import numpy as np

print(dir(data))
print(data.__file__)
print(data.__name__)


class Image:
    # Represents a single image from the dataset.
    # Each image has a label which can be used to identify it uniquely.

    def __init__(
        self,
        label: str,
        json_dict: dict,
    ):
        self.label = label

        # check the json dictionary has the required keys
        required_keys = {"imagePath", "imageData", "imageHeight", "imageWidth"}
        missing_keys = required_keys - json_dict.keys()
        if missing_keys:
            raise ValueError(f"Missing keys: {missing_keys}")

        # extract the values from the json dictionary
        self.image_path = json_dict["imagePath"]
        self.image_data = json_dict["imageData"]
        self.reported_image_height = json_dict["imageHeight"]
        self.reported_image_width = json_dict["imageWidth"]

        # load the image
        self.actual_image = self.load_image_bgr()

        # actual image dimensions
        self.actual_image_height, self.actual_image_width, _ = self.actual_image.shape

        # calculate a scale factor which needs applied to shapes associated with
        # this image to convert from the reported dimensions to the actual dimensions
        self.image_scale = self.calculate_scale()

        # setup padding for the image
        # this creates a padded version of the image (even if padding is 0)
        # and it's that one that is used for extracting patches
        self.padding = 0
        self.pad_image(0)

    def load_image_bgr(
        self,
    ) -> np.ndarray:
        # Load the image from base64 encoded data.
        # we don't use the jpg files

        b64_data = base64.b64decode(self.image_data)
        image = cv2.imdecode(np.frombuffer(b64_data, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Failed to load image data")
        return image

    def calculate_scale(
        self,
    ) -> float:
        # Calculate the scale factor for the image.

        x_scale = self.actual_image_width / self.reported_image_width
        y_scale = self.actual_image_height / self.reported_image_height
        if x_scale != y_scale:
            raise ValueError("Non-uniform scaling not supported")
        return x_scale

    def pad_image(
        self,
        minimum_padding: int = 0,
    ) -> np.ndarray:
        # update the padded image with the given padding size

        if minimum_padding < self.padding:
            return

        self.padding = minimum_padding
        self.padded_image = cv2.copyMakeBorder(
            self.actual_image,
            self.padding,
            self.padding,
            self.padding,
            self.padding,
            cv2.BORDER_WRAP,
        )
        self.padded_image_height, self.padded_image_width, _ = self.padded_image.shape

    def patch(
        self,
        center: np.ndarray,
        size: float,
    ) -> np.ndarray:
        # Extract a patch from the image, centered at (x, y) with the given size.

        x, y = center

        # work out the indexes for the patch
        top = int(y - size / 2)
        left = int(x - size / 2)
        bottom = top + size
        right = left + size

        # work out what padding is needed for the patch
        padding = max(
            -top,
            -left,
            bottom - self.actual_image_height,
            right - self.actual_image_width,
            0,
        )
        self.pad_image(padding)

        # shift the indexes to account for padding
        top += self.padding
        left += self.padding
        bottom += self.padding
        right += self.padding

        patch = self.padded_image[top:bottom, left:right]

        return patch

    def plot(
        self,
        ax: plt.Axes = None,
    ):
        # Display the actual image using the provided axis and setting the correct
        # extent so that the top left corner of the image is at (0, 0).

        if ax is None:
            ax = plt.gca()

        ax.imshow(
            self.actual_image,
            origin="upper",
            extent=[
                0,  # left
                self.actual_image_width,  # right
                self.actual_image_height,  # top
                0,  # bottom
            ],
        )

    def plot_padded(
        self,
        ax: plt.Axes = None,
    ):
        # Display the padded image using the provided axis and setting the correct
        # extent so that the top left corner of the original image (before padding)
        # is at (0, 0).

        if ax is None:
            ax = plt.gca()

        ax.imshow(
            self.padded_image,
            origin="upper",
            extent=[
                -self.padding,  # left
                self.padded_image_width - self.padding,  # right
                self.padded_image_height - self.padding,  # top
                -self.padding,  # bottom
            ],
        )


class Cell:
    # Represents a single cell/RoI in the image.

    def __init__(
        self,
        json_dict: dict,
        image: Image,
    ):
        # check the json dictionary has the required keys
        assert "label" in json_dict
        assert "points" in json_dict

        self.image = image
        self.label = json_dict["label"]

        # we use 1 for mitosis and -1 for non-mitosis
        # 0 is used for unknown

        self.mitosis = 0

        if self.label == "Mitosis":
            self.mitosis = 1

        if self.label == "Non-mitosis":
            self.mitosis = -1

        if self.mitosis == 0 and not self.label.startswith("Blank"):
            raise ValueError(f"Unknown cell label: {self.label}")

        # extract the points for the cells bounding polygon and scale as needed
        points = np.array(json_dict["points"]).reshape(-1, 2)
        points = points * self.image.image_scale

        # store the center and offsets (as integers)
        self.center = np.mean(points, axis=0)
        self.center = np.round(self.center).astype(int)
        self.polygon_offsets = points - self.center
        self.polygon_offsets = np.round(self.polygon_offsets).astype(int) + 1

        self.reset()

    def reset(
        self,
    ):
        self._X = None

        # work out the cell size and scale and store the bounding box points (as offsets)

        max_x = np.max(np.abs(self.polygon_offsets[:, 0]))
        max_y = np.max(np.abs(self.polygon_offsets[:, 1]))

        width = 2 * max_x
        height = 2 * max_y

        self.bounding_box_size = max(width, height)

        self.scale = config.TARGET_CELL_SIZE / self.bounding_box_size
        self.patch_size = np.ceil(config.PATCH_SIZE / self.scale).astype(int)

        # the bounding box offsets

        left = -self.bounding_box_size // 2
        right = left + self.bounding_box_size
        top = -self.bounding_box_size // 2
        bottom = top + self.bounding_box_size

        top_left = np.array([left, top])
        top_right = np.array([right, top])
        bottom_left = np.array([left, bottom])
        bottom_right = np.array([right, bottom])

        self.bounding_box_offsets = np.array(
            [top_left, top_right, bottom_right, bottom_left]
        )

        # the patch box

        left = -self.patch_size // 2
        right = left + self.patch_size
        top = -self.patch_size // 2
        bottom = top + self.patch_size

        top_left = np.array([left, top])
        top_right = np.array([right, top])
        bottom_left = np.array([left, bottom])
        bottom_right = np.array([right, bottom])

        self.patch_box_offsets = np.array(
            [top_left, top_right, bottom_right, bottom_left]
        )

    @property
    def W(
        self,
    ) -> np.ndarray:
        # returns the patch of the image which corresponds to the cell
        # this is the original patch before any augmentation is performed.
        patch = self.image.patch(self.center, self.patch_size)
        #
        # print(self.patch_size, config.PATCH_SIZE, patch.shape)
        patch = cv2.resize(patch, (int(config.PATCH_SIZE), int(config.PATCH_SIZE)))
        # print(patch.shape, config.PATCH_SIZE, patch.shape)
        return patch

    _X: np.ndarray = None

    @property
    def X(
        self,
    ) -> np.ndarray:
        # Returns the patch of the image which corresponds to the cell
        # this version of the patch is extracted from the image and stored
        # in the cell object - it can be manupulated and augmented as needed.

        # if _X is none, we copy the original patch (sopy so we can manipulate it)
        if self._X is None:
            return self.W.copy()

        return self._X

    @X.setter
    def X(
        self,
        value: np.ndarray,
    ):
        # set the X value
        self._X = value

    @property
    def y(
        self,
    ) -> float:
        # Returns the classification of the cell (-1/0/1)
        return self.mitosis

    def plot_polygon(
        self,
        ax: plt.Axes = None,
    ):
        # Display the cells polygon using the provided axis.  the polygon is drawn
        # in the original image coordinates.

        if ax is None:
            ax = plt.gca()

        absolute_points = self.center + self.polygon_offsets
        absolute_points = np.concatenate([absolute_points, [absolute_points[0]]])

        ax.plot(
            absolute_points[:, 0],
            absolute_points[:, 1],
            marker="none",
            linestyle="-",
            linewidth=0.5,
            color="green",
        )

        # plot a fine point at the center

        ax.plot(
            self.center[0],
            self.center[1],
            marker="+",
            markersize=5,
            linestyle="none",
            color="green",
        )

    def plot_bounding_box(
        self,
        ax: plt.Axes = None,
    ):
        # Display the bounding box of the cell using the provided axis.  the box is
        # drawn in the original image coordinates.

        if ax is None:
            ax = plt.gca()

        absolute_points = self.center + self.bounding_box_offsets
        absolute_points = np.concatenate([absolute_points, [absolute_points[0]]])

        ax.plot(
            absolute_points[:, 0],
            absolute_points[:, 1],
            marker="none",
            linestyle="-",
            linewidth=0.5,
            color="blue",
        )

    def plot_patch_box(
        self,
        ax: plt.Axes = None,
    ):
        # Display the patch box of the cell using the provided axis.  the box is
        # drawn in the original image coordinates.

        if ax is None:
            ax = plt.gca()

        absolute_points = self.center + self.patch_box_offsets
        absolute_points = np.concatenate([absolute_points, [absolute_points[0]]])

        ax.plot(
            absolute_points[:, 0],
            absolute_points[:, 1],
            marker="none",
            linestyle="-",
            linewidth=0.5,
            color="yellow",
        )

    def plot_W(
        self,
        ax: plt.Axes,
    ):
        # Display the current patch of the image which corresponds to the cell.

        if ax is None:
            ax = plt.gca()

        width = self.W.shape[0]
        # print(self.W.shape)

        ax.imshow(
            self.W,
            origin="upper",
        )

    def plot_X(
        self,
        ax: plt.Axes,
    ):
        # Display the current patch of the image which corresponds to the cell.

        if ax is None:
            ax = plt.gca()

        width = self.X.shape[0]

        top = self.center[1] - width // 2
        left = self.center[0] - width // 2
        bottom = top + width
        right = left + width

        extent = [left, right, bottom, top]

        ax.imshow(
            self.X,
            origin="upper",
            extent=extent,
        )

        size = config.NETWORK_INPUT_SIZE  # / self.scale
        size = int(size)

        relative_points = np.array(
            [
                [-size // 2, -size // 2],
                [size // 2, -size // 2],
                [size // 2, size // 2],
                [-size // 2, size // 2],
                [-size // 2, -size // 2],
            ]
        )

        absolute_points = self.center + relative_points

        ax.plot(
            absolute_points[:, 0],
            absolute_points[:, 1],
            marker="none",
            linestyle="-",
            linewidth=0.5,
            color="yellow",
        )

        size = config.TARGET_CELL_SIZE  # / self.scale
        size = int(size)

        relative_points = np.array(
            [
                [-size // 2, -size // 2],
                [size // 2, -size // 2],
                [size // 2, size // 2],
                [-size // 2, size // 2],
                [-size // 2, -size // 2],
            ]
        )

        absolute_points = self.center + relative_points

        ax.plot(
            absolute_points[:, 0],
            absolute_points[:, 1],
            marker="none",
            linestyle="-",
            linewidth=0.5,
            color="red",
        )


class CellTrainingDataset(
    data.IndexableDataset,
):
    # Represents a dataset of cells for training.
    # We read all the json files in a directory.  Each json file represents an image
    # with multiple cells.  We extract the cells from the json files and store them
    # in lists of mitotic, non-mitotic cells.
    # by default we shuffle and return a random cell each time we are called.

    def __init__(
        self,
        directory: str,
        shuffle: bool = True,
        validation_fraction: float = 0.2,
    ):
        self.directory = directory
        self.shuffle = shuffle
        self.validation_fraction = validation_fraction

        # load the json files

        json_files = os.listdir(directory)
        json_files = [f for f in json_files if f.endswith(".json")]

        # this list contains the actual cells
        self.cells = []

        # these lists contain the indexes of the different types of cell
        mitotic_cells = []  # self. version will be a numpy array
        non_mitotic_cells = []  # self. version will be a numpy array

        # the list of cells to use for training and validation
        training_cells = []  # self. version will be a numpy array
        validation_cells = []  # self. version will be a numpy array

        for json_file in json_files:

            filename = os.path.join(directory, json_file)
            with open(filename, "r") as f:
                json_dict = json.load(f)

            # get the image label from the filename
            label = json_file.split(".")[0]

            image = Image(label, json_dict)

            for cell_json in json_dict["shapes"]:

                cell = Cell(cell_json, image)

                cell_index = len(self.cells)
                self.cells.append(cell)

                if cell.mitosis == 1.0:
                    mitotic_cells.append(cell_index)
                elif cell.mitosis == -1.0:
                    non_mitotic_cells.append(cell_index)
                else:
                    raise ValueError("Unknown cell type")

        self.mitotic_cells = np.array(mitotic_cells)
        self.non_mitotic_cells = np.array(non_mitotic_cells)

        # split the cells into training and validation sets

        self.num_cells = len(self.cells)
        self.num_mitotic_cells = len(self.mitotic_cells)
        self.num_non_mitotic_cells = len(self.non_mitotic_cells)

        self.num_validation_mitotic_cells = int(
            self.num_mitotic_cells * validation_fraction
        )
        self.num_validation_non_mitotic_cells = int(
            self.num_non_mitotic_cells * validation_fraction
        )
        self.num_validation_cells = (
            self.num_validation_mitotic_cells + self.num_validation_non_mitotic_cells
        )

        self.num_training_mitotic_cells = (
            self.num_mitotic_cells - self.num_validation_mitotic_cells
        )
        self.num_training_non_mitotic_cells = (
            self.num_non_mitotic_cells - self.num_validation_non_mitotic_cells
        )
        self.num_training_cells = (
            self.num_training_mitotic_cells + self.num_training_non_mitotic_cells
        )

        # numpy rng
        rng = np.random.default_rng(42)

        # split the cells randomly
        mitotic_cells = rng.permutation(self.mitotic_cells)
        non_mitotic_cells = rng.permutation(self.non_mitotic_cells)
        self.training_mitotic_cells = mitotic_cells[: self.num_training_mitotic_cells]
        self.validation_mitotic_cells = mitotic_cells[self.num_training_mitotic_cells :]
        self.training_non_mitotic_cells = non_mitotic_cells[
            : self.num_training_non_mitotic_cells
        ]
        self.validation_non_mitotic_cells = non_mitotic_cells[
            self.num_training_non_mitotic_cells :
        ]

        # TODO

    def __len__(
        self,
    ) -> int:
        return len(self.cells)

    def reset(
        self,
    ):
        # reset the cells to their original state
        for cell in self.cells:
            cell.reset()

    def __getitem__(
        self,
        index: int,
    ) -> Cell:
        return self.cells[index]

    @property
    def training_split(
        self,
    ) -> data.SelectiveIndexableDataset:

        # this will return a random cell each time it is called
        training_cells = np.concatenate(
            [self.training_mitotic_cells, self.training_non_mitotic_cells]
        )

        dataset = data.SelectiveIndexableDataset(
            source=self.cells,
            selection=training_cells,
        )

        return dataset

    @property
    def validation_split(
        self,
    ) -> data.SelectiveIndexableDataset:

        # this will return a random cell each time it is called
        validation_cells = np.concatenate(
            [self.validation_mitotic_cells, self.validation_non_mitotic_cells]
        )

        dataset = data.SelectiveIndexableDataset(
            source=self.cells,
            selection=validation_cells,
        )

        return dataset


class CellTestDataset(
    data.IndexableDataset,
):
    # Represents a dataset of cells for testing.
    # We read all the json files in a directory.  Each json file represents an image
    # with multiple cells.  We extract the cells from the json files and store them.
    # we don't shuffle.

    def __init__(
        self,
        directory: str,
    ):
        super().__init__()

        self.directory = directory

        # load the json files

        json_files = os.listdir(directory)
        json_files = [f for f in json_files if f.endswith(".json")]

        # this list contains the actual cells
        self.cells = []

        for json_file in json_files:

            with open(f"{directory}/{json_file}", "r") as f:
                json_dict = json.load(f)

            # get the image label from the filename
            label = json_file.split(".")[0]

            image = Image(label, json_dict)

            for cell_json in json_dict["shapes"]:

                cell = Cell(cell_json, image)

                self.cells.append(cell)

    def __len__(
        self,
    ) -> int:
        return len(self.cells)

    # this is a tensorflow generator for the dataset
    def __getitem__(
        self,
        index: int,
    ):
        return self.cells[index]

    def reset(
        self,
    ):
        return
        # reset the cells to their original state
        for cell in self.cells:
            cell.reset()
