from .main import video_to_faces


def image_gallery(*args, **kwargs):
    from .utils.gallery import image_gallery as show_gallery
    return show_gallery(*args, **kwargs)


def dataframe_with_images(*args, **kwargs):
    from .utils.gallery import dataframe_with_images as show_dataframe
    return show_dataframe(*args, **kwargs)
