from torchvision import datasets, transforms

def get_dataset(name="mnist", root="data", train=True):
    tf = transforms.Compose([
        transforms.ToTensor(),                 # PIL uint8 [0,255] HWC -> float32 [0,1] CHW, (1,28,28)
        transforms.Pad(2),                     # -> (1,32,32); fill=0 is black, matches background
        transforms.Normalize((0.5,), (0.5,)),  # (x - 0.5) / 0.5  ->  [-1,1]
    ])

    if name == "mnist":
        train_ds = datasets.MNIST(root=root, train=train, download=True, transform=tf)
    elif name == "fashion":
        train_ds = datasets.FashionMNIST(root=root, train=train, download=True, transform=tf)
    else:
        raise ValueError(f"unknown dataset: {name}")

    return train_ds
