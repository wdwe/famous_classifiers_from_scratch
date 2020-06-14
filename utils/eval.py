import numpy as np
import torch
import torchvision

def accuracy(probabilities, labels, topk = (1, )):
    """return the accuracy for top k"""
    num_images = labels.size()[0]
    max_k = max(topk)
    _, top_cat = probabilities.topk(max_k)
    correct = top_cat == labels
    acc = [torch.sum(correct[:, :k]).type(torch.float32) / num_images for k in topk]
    return acc


if __name__ == "__main__":
    import sys
    import os
    sys.path.append("..")
    from data.dataset import EvalDataset
    from torch.utils.data import DataLoader
    import torch.nn as nn
    from utils import AverageMeter
    from tqdm import tqdm

    alexnet = torchvision.models.alexnet(pretrained = True)

    fname = True
    imagenet_dir = "../../datasets/imagenet/ILSVRC2015/Data/CLS-LOC/"
    val_dir = os.path.join(imagenet_dir, "val")
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    alexnet_data_config = {
        "imdir" : val_dir, 
        "mean" : mean, 
        "std" : std,
        "rescale_sizes" : [256],
        "center_square" : True,
        "crop" : "fivecrop",
        "horizontal_flip" : True,
        "fname" : fname
    }

    eval_dataset = EvalDataset(**alexnet_data_config)
    eval_loader = DataLoader(eval_dataset, batch_size = 32)

    top1_meter = AverageMeter()
    top5_meter = AverageMeter()
    softmax = nn.Softmax(-1)
    with torch.no_grad():
        for i, data in enumerate(tqdm(eval_loader)):
            images = data['image']
            labels = data['label']
            # remember our dataset already returns a 4D tensor of [ncorp, c, h, w]
            # dataloader will add a new bs dimension
            bs, ncrops, c, h, w = images.size()
            images = images.view((-1, c, h, w))
            labels = labels.view((bs, 1))
            result = alexnet(images)
            prob = softmax(result)
            prob = prob.view((bs, ncrops, -1))
            prob = torch.mean(prob, dim = 1)
            acc = accuracy(prob, labels, topk = (1, 5))
            top1_meter.update(acc[0].item(), n = bs)
            top5_meter.update(acc[1].item(), n = bs)
            

    print(f"Top 1 accuracy: {top1_meter.avg}")
    print(f"Top 5 accuracy: {top5_meter.avg}")
    # with open("logs.txt", "w+") as f:
    #     f.write(f"Top 1 accuracy: {top1_meter.avg}\n")
    #     f.write(f"Top 5 accuracy: {top5_meter.avg}")