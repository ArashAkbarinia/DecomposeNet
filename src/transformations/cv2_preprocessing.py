"""
Preparing the input image to be inputted to a network.
"""

import numpy as np

import cv2

from . import colour_spaces
from . import normalisations


class ColourTransformation(object):

    def __init__(self, colour_space='rgb'):
        self.colour_space = colour_space

    def __call__(self, img):
        if self.colour_space != 'rgb':
            img = np.asarray(img).copy()
            if self.colour_space == 'lab':
                img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
            elif self.colour_space == 'labhue':
                img_hue = colour_spaces.lab2lch01(
                    colour_spaces.rgb2opponency(img.copy(), 'lab')
                )
                img_hue = normalisations.uint8im(img_hue)
                img_lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
                img = np.concatenate([img_lab, img_hue[:, :, 2:3]], axis=2)
            elif self.colour_space == 'dkl':
                img = colour_spaces.rgb2dkl01(img)
                img = normalisations.uint8im(img)
            elif self.colour_space == 'hsv':
                img = colour_spaces.rgb2hsv01(img)
                img = normalisations.uint8im(img)
            elif self.colour_space == 'lms':
                img = colour_spaces.rgb2lms01(img)
                img = normalisations.uint8im(img)
            elif self.colour_space == 'yog':
                img = colour_spaces.rgb2yog01(img)
                img = normalisations.uint8im(img)

        return img
