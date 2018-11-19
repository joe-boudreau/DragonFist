import os

def ensure_is_plottable(image_array):
    """If the passed image or set of images has a channel count of 1,
       remove the channel dimension so matplotlib can plot it."""
    if image_array.shape[-1] == 1:
        return image_array.reshape(image_array.shape[:-1])
    else:
        return image_array


def create_image_folder(save_image_location, category_name):
    if save_image_location != None and save_image_location != '':
        save_image_location += '/' + category_name
        os.makedirs(save_image_location, exist_ok=True)
    return save_image_location
