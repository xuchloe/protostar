# %%
from casatools import (image as IA)
# %%
def get_region_stats(image_file: str, include: bool, center_x: float, center_y: float, r1: float, r2: float):
    '''Given an image, center x and y positions (degrees), inner radius (arcsec), whether to include or exclude region inside the inner radius,
    and optionally given an outer radius (arcsec), return a dictionary with the maximum flux and rms in the specified region'''

    ia = IA()
    ia.open(image_file)

    if include == True:
        region = f'circle[[{center_x}deg, {center_y}deg], {r1}arcsec]'
    elif include == False:
        region = f'annulus[[{center_x}deg, {center_y}deg], [{r1}arcsec, {r2}arcsec]]'

    sub_image = ia.subimage(region = region)
    sub_stats = sub_image.statistics()

    ia.close()

    maximum = sub_stats['max'][0]
    rms = sub_stats['rms'][0]

    return {'maximum': float(maximum), 'rms': float(rms)}
