import numpy as np
import torch
import torch.nn as nn
import torchvision
from utils import AverageMeter
from tqdm import tqdm


def accuracy(probabilities, labels, topk = (1, )):
    """return the accuracy for top k"""
    num_images = labels.size()[0]
    max_k = max(topk)
    _, top_cat = probabilities.topk(max_k)
    correct = top_cat == labels
    accu = [torch.sum(correct[:, :k]).type(torch.float32) / num_images for k in topk]
    return accu


def evaluate(model, dataloader, topk = (1, ), verbose = False):

    model.eval()
    accu_meters = [AverageMeter() for _ in topk]
    softmax = nn.Softmax(-1)

    if torch.cuda.is_available():
        model = model.cuda()

    with torch.no_grad():
        for i, data in enumerate(tqdm(eval_loader)):
            images = data['image']
            labels = data['label']
            if torch.cuda.is_available():
                images = images.cuda()
                labels = labels.cuda()
            # remember our dataset already returns a 4D tensor of [ncorp, c, h, w]
            # dataloader will add a new bs dimension
            bs, ncrops, c, h, w = images.size()
            images = images.view((-1, c, h, w))
            labels = labels.view((bs, 1))
            result = model(images)
            prob = softmax(result)
            prob = prob.view((bs, ncrops, -1))
            prob = torch.mean(prob, dim = 1)
            accu = accuracy(prob, labels, topk = (1, 5))
            # update accu_meters
            for i, meter in enumerate(accu_meters):
                meter.update(accu[i].item(), n = bs)

    if verbose:
        for i, meter in enumerate(accu_meters):
            print(f"Top {topk[i]} accuracy: {meter.avg}")


if __name__ == "__main__":
    import sys
    import os
    sys.path.append("..")
    from data.dataset import EvalDataset
    from torch.utils.data import DataLoader

    model = torchvision.models.alexnet(pretrained=True)

    imagenet_dir = "../../datasets/imagenet/ILSVRC2015/Data/CLS-LOC/"
    val_dir = os.path.join(imagenet_dir, "val")
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    dataset_config = {
        "imdir" : val_dir, 
        "mean" : mean, 
        "std" : std,
        "rescale_sizes" : [256],
        "center_square" : True,
        "crop" : "center",
        "horizontal_flip" : False,
        "fname" : False
    }


    eval_dataset = EvalDataset(**dataset_config)
    eval_loader = DataLoader(eval_dataset, batch_size = 512, num_workers = 8)
    
    evaluate(model, eval_loader, topk = (1, 5), verbose = True)

    # alexnet 1 center crop
    # Top 1 accuracy: 0.5651800000572205
    # Top 5 accuracy: 0.7906999998283386

    # vgg16 1 center crop
    # Top 1 accuracy: 0.7159200002861023
    # Top 5 accuracy: 0.9038200002670288
