import cv2
import numpy as np
import os

from PIL import  Image


class Preprocessor:

    directory: str
    files: list

    def __init__(self, directory):
        self.directory = directory
        if not os.path.isdir(self.directory):
            raise Exception(f'Path is not a directory, path: \'{directory}\'')
        else:
            self.files = os.listdir(directory)

    def crop_images_in_directory(self):

        for file in self.files:
            self._crop_image(self.directory+f'/{file}')

    def _crop_image(self, img_path):
        img = cv2.imread(img_path)
        if img is not None:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # Find contours and hierarchy, use RETR_TREE for creating a tree of contours within contours
            cnts, hiers = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[
                          -2:]  # [-2:] indexing takes return value before last (due to OpenCV compatibility issues).

            parent = hiers[0, :, 3]

            hist = np.bincount(np.maximum(parent, 0))
            max_n_childs_idx = hist.argmax()

            # Get the contour with the maximum child contours
            c = cnts[max_n_childs_idx]

            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(c)

            # Crop the bounding rectangle out of img
            img = img[y:y + h, x:x + w, :]
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Convert to binary image (after cropping) and invert polarity
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

            # cv2.imshow('thresh', thresh);cv2.waitKey(0);cv2.destroyAllWindows()

            # Find connected components (clusters)
            nlabel, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity=8)

            # Delete large, small, tall and wide clusters - not letters for sure
            max_area = 2000
            min_area = 10
            max_width = 100
            max_height = 100
            for i in range(1, nlabel):
                if (stats[i, cv2.CC_STAT_AREA] > max_area) or \
                        (stats[i, cv2.CC_STAT_AREA] < min_area) or \
                        (stats[i, cv2.CC_STAT_WIDTH] > max_width) or \
                        (stats[i, cv2.CC_STAT_HEIGHT] > max_height):
                    thresh[labels == i] = 0

            # cv2.imshow('thresh', thresh);cv2.waitKey(0);cv2.destroyAllWindows()

            # Use "closing" morphological operation for uniting text area
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, np.ones((50, 50)))

            # Find contours once more
            cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2]

            # Get contour with maximum area
            if cnts is not None and len(cnts) != 0:
                c = max(cnts, key=cv2.contourArea)

                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(c)

                # Crop the bounding rectangle out of img (leave some margins)
                roi = img[y - 5:y + h + 5, x - 5:x + w + 5]
                try:
                    extension = img_path.split('.')
                    cv2.imwrite(img_path+'_preprocessed.'+extension[len(extension)-1], roi)
                except Exception as e:
                    print('Error imagen: ', img_path)

preprocessor = Preprocessor('test_images')
preprocessor.crop_images_in_directory()
