import torch
import torch.nn as nn
import timm

class ChestXRayClassifier(nn.Module):
    def __init__(self, backbone_name="densenet121", num_classes=4, pretrained=True, drop_rate=0.2):
        super().__init__()
        self.backbone_name = backbone_name
        self.num_classes = num_classes
        
        # Instantiate backbone using timm
        self.model = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=num_classes,
            drop_rate=drop_rate
        )

    def forward(self, x):
        return self.model(x)

def build_model(backbone_name="densenet121", num_classes=4, pretrained=True, drop_rate=0.2):
    return ChestXRayClassifier(
        backbone_name=backbone_name,
        num_classes=num_classes,
        pretrained=pretrained,
        drop_rate=drop_rate
    )
