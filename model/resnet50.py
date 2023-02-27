import torch
from torch import nn
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models._utils import IntermediateLayerGetter


class ResNet50(nn.Module):

    def __init__(self, dropout):
        """
        The aim of this class is to modify the forward method, for extracting the embedding features
        Without having to modify the original class
        This network expects input size of 3 x 160 x 160
        For the Binary image reconstructing, it will have size 1 x 64 x 64
        """
        super().__init__()

        backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        returned_layers = [4]
        return_layers = {f'layer{k}': str(v) for v, k in enumerate(returned_layers)}
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 128)
        )

    def forward(self, x):
        x = self.body(x)
        x = self.avgpool(x['0'])
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


if __name__ == '__main__':
    from torchinfo import summary
    device = torch.device('cpu')
    model = ResNet50(dropout=0.5)
    batch_size = 10
    summary(model, input_size=(batch_size, 3, 224, 224), device=device)
