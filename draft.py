from lib.dataset import get_cifar10_loader

if __name__ == "__main__":
    tq, vq = get_cifar10_loader("./data", 64)
    print(len(tq), next(iter(tq))[0].shape)
