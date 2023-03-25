import torch
import torchvision
from collections import OrderedDict

m = torchvision.ops.MultiScaleRoIAlign(['feat1', 'feat3'], 3, 1)
i = OrderedDict()
i['feat1'] = torch.rand(1, 5, 64, 64)
i['feat2'] = torch.rand(1, 5, 32, 32)  # this feature won't be used in the pooling
i['feat3'] = torch.rand(1, 5, 16, 16)
# create some random bounding boxes
boxes = torch.rand(6, 4) * 256; boxes[:, 2:] += boxes[:, :2]
# original image size, before computing the feature maps
image_sizes = [(512, 512)]
output = m(i, [boxes], image_sizes)
print(output.shape)