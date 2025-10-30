import torch
import timm

model = timm.create_model(
    'darknet53.c2ns_in1k', 
    pretrained=True, 
    features_only=True
)
model.eval()
with torch.no_grad():
    x = torch.randn(1, 3, 416, 416)
    features = model(x)
    for i, feature in enumerate(features):
        print(f"Feature {i}: shape = {feature.shape}")