import torch
import torchvision
from transform import get_transform
from dataset import PennFudanDataset
import utils

if __name__ == "__main__":
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    dataset = PennFudanDataset('/Users/shubhamrandive/Documents/datasets/PennFudanPed', get_transform(train=True))

    dataloader =  torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True, num_workers=4, collate_fn=utils.collate_fn)

    images, targets = next(iter(dataloader))

    images = list(image for image in images)

    targets = [{k: v for k, v in t.items()} for t in targets]

    output = model(images, targets)

    model.eval()

    x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]

    predictions = model(x)

    print(predictions)