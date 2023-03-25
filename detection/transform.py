
import torchvision.transforms as transforms
import torch

def get_transform(train):
    transform_list = []
    transform_list.append(transforms.ToTensor())
    transform_list.append(transforms.ConvertImageDtype(torch.float32))
    if train:
        transform_list.append(transforms.RandomHorizontalFlip(p=0.5))
    return transforms.Compose(transform_list)



if __name__ == "__main__":

    transform = get_transform(True)

    # create a random image tensor
    img = torch.rand( 3, 300, 400)

    #convert to PIL image using torchvision
    import torchvision
    img = torchvision.transforms.ToPILImage()(img)

    #apply transform
    img = transform(img)

    print(img.shape)

