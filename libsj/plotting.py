__author__ = "Jakub Sochor"
__copyright__ = "Copyright 2018, Jakub Sochor"
__license__ = "MIT"


import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import numpy as np


#%%
def plot_3DBB(bb3d, ax=None, everything = False, *args, **kwargs):
    if ax is None:
        ax = plt.gca()
    if everything:
        for i in (3,4,6):
            ax.plot(bb3d[[i,7], 0], bb3d[[i,7], 1], ls="--", *args, **kwargs)
    ax.plot(bb3d[[0,1,2,3,0], 0], bb3d[[0,1,2,3,0], 1], *args, **kwargs)
    ax.plot(bb3d[[4,5,6], 0], bb3d[[4,5,6], 1], *args, **kwargs)
    for i in range(3):
        ax.plot(bb3d[[i,i+4], 0], bb3d[[i,i+4], 1], *args, **kwargs)


#%%
def plot_2DBB(bb2d, ax=None, *args, **kwargs):
    if ax is None:
        ax = plt.gca()
    ax.add_patch(patches.Rectangle(tuple(bb2d[0:2]), bb2d[2], bb2d[3], fill=False, *args, **kwargs))

    
    
    
    
def cv_draw_text(frame, text, position, font = cv2.FONT_HERSHEY_DUPLEX, font_scale = 1, font_thickness = 2,
                 padding = 5, text_color = (0, 0, 0), background_color = (255, 255, 102)):
    size, baseline = cv2.getTextSize(text, font, font_scale, font_thickness)
    x,y = position
    y1,y2 = np.clip([y - size[1] - padding, y + padding], 0, frame.shape[0]).astype(int)
    x1,x2 = np.clip([x - padding, x + size[0] + padding], 0, frame.shape[1]).astype(int)
    frame[y1:y2, x1:x2, :] = background_color
    cv2.putText(frame, text, (x, y), font, font_scale, text_color, font_thickness, cv2.LINE_AA)