import torch
from torch import nn


class BasicClassifyHead(nn.Module):

    def __init__(self, in_channel, num_classes):
        super(BasicClassifyHead, self).__init__()

        self.features = nn.Sequential(
            nn.ReLU(inplace=True),  # assuming input size 8x8
            # nn.AvgPool2d(5, stride=3, padding=0, count_include_pad=False),  # image size = 2 x 2
            nn.Conv2d(in_channel, 256, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            # nn.Conv2d(256, 768, 2, bias=False),
            # nn.BatchNorm2d(768),
            # nn.ReLU(inplace=True)
        )
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x.view(x.size(0), -1))
        return x


if __name__ == "__main__":
    from lib.basic import BasicRebuildNetwork
    from lib.basic.genotypes import DARTS_V1

    dummy = torch.rand(7, 3, 32, 32).cuda()
    model = BasicRebuildNetwork(DARTS_V1).cuda()
    head = BasicClassifyHead(model.out_channel, 10).cuda()

    output = head(model(dummy))
    print(output.shape)
