import torch
import PIL
import torchvision.transforms as tsfm
from torchvision.datasets import ImageFolder as ImageFolder


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
                resized_images.append(tsfm.functional.resize(image, size))

        return resized_images

class CenterCropMultiple:
    def __call__(self, images):
        if not isinstance(images, list):
            images = [images]
        crops = [tsfm.CenterCrop(min(image.size))(image) for image in images]

        return crops

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
    """Callable class for image transform
    its __call__ takes a PIL image or a list of PIL images and returns
    a list of 5 crops and the image resized to crop size.
    """
    def __init__(self, size):
        self.size = size

    def __call__(self, images):
        if not isinstance(images, list):
            images = [images]
        crops = []
        for image in images:
            assert image.width == image.height, "image is not square"
            crops.extend(tsfm.FiveCrop(self.size)(image))
            crops.append(tsfm.functional.resize(image, self.size))

        return crops


class ThreeCropMultiple:
    def __call__(self, images):
        if not isinstance(images, list):
            images = [images]
        crops = []
        for image in images:
            if image.width > image.height:
                # return left, center, right crops
                size = image.height
                stride = (image.width - size) / 3
                for step in range(3):
                    left = int(step * stride)
                    crops.append(image.crop((left, 0, left + size, size)))
            else:
                # return top, center, right crops
                size = image.width
                stride = (image.height - size) / 3
                for step in range(3):
                    top = int(step * stride)
                    crops.append(image.crop((0, top, size, top + size)))

        return crops


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


class HorizontalFlipMultiple:
    def __call__(self, images):
        if not isinstance(images, list):
            images = [images]
        flipped = []
        for image in images:
            flipped.append(PIL.ImageOps.mirror(image))
        return images + flipped

class ToTensorMultiple:
    """Callable class for image transform
    Convert a list of PIL images to a list of tensors
    """
    def __call__(self, images):
        if not isinstance(images, list):
            images = [images]
        images = [tsfm.ToTensor()(image) for image in images]

        return images


class NormalizeMultiple:
    def __init__(self, 
        mean = [0.485, 0.456, 0.406], 
        std = [0.229, 0.224, 0.225]):
        self.mean = mean
        self.std = std

    def __call__(self, tensor_images):
        if not isinstance(tensor_images, list):
            tensor_images = [tensor_images]
        return [tsfm.Normalize(self.mean, self.std)(image) for image in tensor_images]


class EvalDataset(ImageFolder):
    def __init__(
        self,
        imdir,
        input_size = 224,
        rescale_sizes = [256],
        center_square = False,
        crop = "fivecrop",
        mean = [0.485, 0.456, 0.406],
        std = [0.229, 0.224, 0.225],
        horizontal_flip = True
    ):
        """Initialise the class

        Args:
            imdir (str): The root directory for all images
            input_size (int): The size of network input.
                If crop argument is None, then this argument is ignored.
            rescale_sizes (list): The elements can be either int or tuple.
                Tuple defines the exact size the image to be resized to before cropping.
                Int defines the size the shorter image dimension to be resized to.
                (default is False)
            center_square (bool): Whether to take the center square of the image.
            crop (str, None): Must be of "fivecrop", "gridcrop", "inception" or None.
                If this argument is None, the rescaled images are returned without cropping.
            horizontal_flip (bool): Whether to make copies of the crops' horizontal flip.
            mean (list): The RGB pixel mean for normalization.
            std (list): The RGB pixel standard deviation for normalization.
        Returns:
            None
        """

        self.imdir = imdir
        self.input_size =  input_size
        self.rescale_sizes = rescale_sizes
        assert crop in ["fivecrop", "inception", "gridcrop", None], "crop can only be one of ['alex', 'inception', 'vgg', None]"
        self.center_square = center_square
        self.crop = crop
        self.horizontal_flip = horizontal_flip
        self.mean = mean
        self.std = std
        transforms = self._get_transforms()
        super().__init__(root = self.imdir, transform = transforms)


    def _get_transforms(self):
        transforms = []

        
        transforms.append(ResizeMultiple(self.rescale_sizes))

        if self.center_square:
            transforms.append(CenterCropMultiple())

        if self.crop == "fivecrop":
            transforms.append(FiveCropMultiple(self.input_size))
        elif self.crop == "gridcrop":
            transforms.append(GridCropMultiple(self.input_size))
        elif self.crop == "inception":
            transforms.append(ThreeCropMultiple())
            transforms.append(FiveAndOneCropMultiple(self.input_size))

        if self.horizontal_flip:
            transforms.append(HorizontalFlipMultiple())

        transforms.append(ToTensorMultiple())
        transforms.append(NormalizeMultiple(self.mean, self.std))
        # convert the list of tensors to a 4-D tensor of dims
        # [Num_crops, channels, H, W]
        transforms.append(tsfm.Lambda(lambda crops: torch.stack(crops)))
        transforms = tsfm.Compose(transforms)

        return transforms


if __name__ == "__main__":
    from tensorboardX import SummaryWriter
    writer = SummaryWriter("logs/dataset/")
    dataset_dir = "../datasets/imagenet/ILSVRC2015/Data/CLS-LOC/val"
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    dataset = EvalDataset(
        imdir = dataset_dir, 
        mean = mean, 
        std = std,
        rescale_sizes = [256, 384, 512],
        center_square = True,
        crop = "gridcrop"
        )
    
    print(dataset[0][0].shape)