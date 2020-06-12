import pytorch
from PIL import Image as PImage
import torchvision.transforms as tsfm
import torchvision.datasets.ImageFolder as ImageFolder


class ResizeMultiple:
    """Callable class for image transform
    its __call__ takes a PIL image or a list of PIL images and returns
    a list of resized images.
    """
    def __init__(self, rescale_sizes):
        self.sizes = rescale_sizes

    def __call__(self, images):
        if not isinstance(images, list):
            images = [images]
        resized_images = []
        for size in self.sizes:
            for image in images:
                resized_images.append(tsfm.functional.resize(image, sizes))

        return resized_images


class FiveCropMultiple:
    def __init__(self, size):
        self.size = size

    def __call__(self, images):
        if not isinstance(images, list):
            images = [images]
        crops = []
        for image in images:
            crops.extend(tsfm.FiveCrop(self.size)(image))
        
        return crops

class FiveAndOneCropMultiple:
    def __init__(self, size):
        self.size = size
    
    def __call__(self, images):
        

class GridCropMultiple:
    def __init__(self, size, steps = 5):
        self.size = size
        self.steps = steps

    def __call__(self, images):
        if not isinstance(images, list):
            images = [images]
        crops = []
        for image in images:
            h = image.height
            w = image.width
            v_stride = (h - self.size) / self.steps
            h_stride = (w - self.size) / self.steps
            for h_step in range(self.steps):
                for v_step in range(self.steps):
                    left = int(h_stride * h_step)
                    top = int(v_stride * v_step)
                    box = (left, top, left + self.size , top + self.size)
                    crops.append(image.crop(box))

        return crops
    


class EvalDataset(ImageFolder):
    def __init__(
        self,
        imdir,
        input_size = 224,
        rescale_sizes = [(256, 256)],
        crop = "alex",
        horizontal_flip = True
    ):
        """Initialise the class

        Args:
            imdir (str): The root directory for all images
            input_size (int or None): The size of network input.
                If input_size is None, then image is not cropped.
            rescale_sizes (list): The elements can be either int or tuple. 
                Tuple defines the exact size the image to be resized to before cropping.
                Int defines the size the shorter image dimension to be resized to.
                (default is False)
            crop (str): Must be of "alex", "vgg" or "inception". 
                this argument is ignored if input_size is None. i.e. The rescaled images are returned without cropping.
            horizontal_flip (bool): Whether to make copies of the crops' horizontal flip.

        Returns:
            None
        """

        self.im_dir = im_dir
        self.input_size =  input_size
        self.rescale_sizes = rescale_sizes
        assert crop in ["alex", "inception", "vgg", None], "crop can only be one of ['alex', 'inception', 'vgg', None]"
        self.crop = crop
        self.flip = horizontal_flip
    
    

    def _get_transforms(self):
        transforms = []
        transforms.append(self.ResizeMultiple(self.rescale_sizes))

