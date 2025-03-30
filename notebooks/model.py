import torch.nn as nn
import torch.nn.functional as F

class SkinDiseaseCNN(nn.Module):
    def __init__(self, num_classes=9):
        super(SkinDiseaseCNN, self).__init__()
        
        # Feature Extraction
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.5)  # Prevent overfitting
        
        # Adaptive Pooling to make it more flexible
        self.global_pool = nn.AdaptiveAvgPool2d((4, 4))
        
        # Fully Connected Layers
        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(F.leaky_relu(self.bn1(self.conv1(x))))
        x = self.pool(F.leaky_relu(self.bn2(self.conv2(x))))
        x = self.pool(F.leaky_relu(self.bn3(self.conv3(x))))
        x = self.pool(F.leaky_relu(self.bn4(self.conv4(x))))
        
        x = self.global_pool(x)  # Adaptive pooling
        x = x.view(x.size(0), -1)  # Flatten
        
        x = F.leaky_relu(self.fc1(x))
        x = self.dropout(x)  # Dropout for regularization
        x = F.leaky_relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)  # Final classification layer
        
        return x

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torchvision.models as models

# class UNetDecoder(nn.Module):
#     """Decoder module for UNet with upsampling layers."""
#     def __init__(self, in_channels, out_channels):
#         super(UNetDecoder, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
#         self.bn1 = nn.BatchNorm2d(out_channels)
#         self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
#         self.bn2 = nn.BatchNorm2d(out_channels)
#         self.up = nn.ConvTranspose2d(out_channels, out_channels // 2, kernel_size=2, stride=2)

#     def forward(self, x, skip_connection):
#         x = F.leaky_relu(self.bn1(self.conv1(x)))
#         x = F.leaky_relu(self.bn2(self.conv2(x)))
#         x = self.up(x)
        
#         # Resize skip connection to match
#         if x.shape != skip_connection.shape:
#             skip_connection = F.interpolate(skip_connection, size=x.shape[2:], mode='bilinear', align_corners=True)

#         x = torch.cat([x, skip_connection], dim=1)  # Concatenation after resizing
#         return x

# class UNet(nn.Module):
#     def __init__(self, num_classes=9):
#         super(UNet, self).__init__()

#         # Load EfficientNet-B0
#         self.encoder = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)

#         # Freeze the encoder layers
#         for param in self.encoder.features.parameters():
#             param.requires_grad = False

#         # Extract feature maps from different levels for skip connections
#         self.enc1 = self.encoder.features[:2]  # First few layers (input -> 24 channels)
#         self.enc2 = self.encoder.features[2:4]  # Downsample -> 40 channels
#         self.enc3 = self.encoder.features[4:6]  # Downsample -> 80 channels
#         self.enc4 = self.encoder.features[6:8]  # Downsample -> 112 channels
#         self.enc5 = self.encoder.features[8]    # Last layer -> 1280 channels

#         # Decoder
#         self.dec4 = UNetDecoder(1280, 512)
#         self.dec3 = UNetDecoder(512, 256)  # Skip connection with 112 channels
#         self.dec2 = UNetDecoder(256, 128)  # Skip connection with 80 channels
#         self.dec1 = UNetDecoder(128, 64)   # Skip connection with 40 channels

#         # Final output layer
#         self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)

#     def forward(self, x):
#         # Encoder forward pass
#         x1 = self.enc1(x)  # 24 channels
#         x2 = self.enc2(x1)  # 40 channels
#         x3 = self.enc3(x2)  # 80 channels
#         x4 = self.enc4(x3)  # 112 channels
#         x5 = self.enc5(x4)  # 1280 channels

#         # Decoder with skip connections
#         d4 = self.dec4(x5, x4)  # 1280 -> 512
#         d3 = self.dec3(d4, x3)  # 512 -> 256
#         d2 = self.dec2(d3, x2)  # 256 -> 128
#         d1 = self.dec1(d2, x1)  # 128 -> 64

#         # Final segmentation/classification layer
#         out = self.final_conv(d1)  # Output with num_classes channels

#         return out

# Example Usage:
# if __name__ == "__main__":
#     model = UNetEfficientNet(num_classes=9)
#     model = model.to("cuda" if torch.cuda.is_available() else "cpu")

#     # Test with a dummy input
#     sample_input = torch.randn(1, 3, 256, 256).to("cuda" if torch.cuda.is_available() else "cpu")
#     output = model(sample_input)
#     print("Output shape:", output.shape)  # Expected: (1, 9, 256, 256)







# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from timm import create_model

# class SkinDiseaseSwin(nn.Module):
#     def __init__(self, num_classes=9):
#         super(SkinDiseaseSwin, self).__init__()

#         # Load Swin Transformer model
#         self.swin = create_model("swin_tiny_patch4_window7_224", pretrained=True, num_classes=0)  # Remove default head

#         # Feature dimension of Swin Transformer
#         self.feature_dim = self.swin.num_features  # Typically 768 for Swin-Tiny
        
#         # Add a new classification head
#         self.head = nn.Linear(self.feature_dim, num_classes)
        
#         # Dropout for regularization
#         self.dropout = nn.Dropout(0.5)

#     def forward(self, x):
#         # Resize input to match Swin Transformer requirements (224x224)
#         x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)

#         # Forward pass through Swin Transformer
#         x = self.swin(x)  # Shape: (batch_size, 7, 7, 768) (assuming 7x7 patches)

#         # Flatten correctly
#         x = x.view(x.shape[0], -1, x.shape[-1])  # Shape: (batch_size, num_patches, feature_dim)

#         # Global Average Pooling (correct way)
#         x = x.mean(dim=1)  # Shape: (batch_size, 768) (Correct)

#         # Ensure it's 2D before passing into the classifier
#         x = x.view(x.shape[0], -1)  

#         # Dropout and Classification
#         x = self.dropout(x)
#         x = self.head(x)  # Should now be (batch_size, num_classes)

#         return x




# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from timm import create_model

# class HybridSkinDiseaseModel(nn.Module):
#     def __init__(self, num_classes=9):
#         super(HybridSkinDiseaseModel, self).__init__()

#         #### 🔹 CNN Feature Extractor (Local Features)
#         self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
#         self.bn1 = nn.BatchNorm2d(32)
        
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
#         self.bn2 = nn.BatchNorm2d(64)
        
#         self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
#         self.bn3 = nn.BatchNorm2d(128)
        
#         self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.global_pool = nn.AdaptiveAvgPool2d((4, 4))

#         #### 🔹 Swin Transformer (Global Features)
#         self.swin = create_model("swin_tiny_patch4_window7_224", pretrained=True, num_classes=0)
#         self.feature_dim = self.swin.num_features  # 768 for Swin-Tiny

#         #### 🔹 Fully Connected Layers
#         self.fc_cnn = nn.Linear(128 * 4 * 4, 256)  # CNN output
#         self.fc_swin = nn.Linear(self.feature_dim, 256)  # Swin output
        
#         self.fc_final = nn.Linear(512, num_classes)  # Merged output layer
#         self.dropout = nn.Dropout(0.5)  # Regularization

#     def forward(self, x):
#         #### 🔥 CNN Forward Pass
#         x_cnn = self.pool(F.leaky_relu(self.bn1(self.conv1(x))))
#         x_cnn = self.pool(F.leaky_relu(self.bn2(self.conv2(x_cnn))))
#         x_cnn = self.pool(F.leaky_relu(self.bn3(self.conv3(x_cnn))))
#         x_cnn = self.global_pool(x_cnn)  
#         x_cnn = x_cnn.view(x_cnn.size(0), -1)  # Flatten
#         x_cnn = F.leaky_relu(self.fc_cnn(x_cnn))

#         #### 🔥 Swin Transformer Forward Pass
#         x_swin = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
#         x_swin = self.swin(x_swin)  # (batch_size, feature_dim)
#         x_swin = F.leaky_relu(self.fc_swin(x_swin))

#         #### 🔥 Fusion of CNN & Swin Transformer
#         x_combined = torch.cat((x_cnn, x_swin), dim=1)  # Merge both feature vectors
#         x_combined = self.dropout(x_combined)
#         x_out = self.fc_final(x_combined)  # Final classification layer

#         return x_out




# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class HybridSkinDiseaseModel(nn.Module):
#     def __init__(self, num_classes):
#         super(HybridSkinDiseaseModel, self).__init__()
#         self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.fc1 = nn.Linear(64 * 56 * 56, 512)
#         self.fc2 = nn.Linear(512, num_classes)
#         self.unet_fc = nn.Linear(25690112, 256)  # Fix incorrect size if needed

#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))

#         x_unet = x.view(x.size(0), -1)  # Flatten before FC layer
#         print(f"x_unet shape before FC: {x_unet.shape}")  # Debugging

#         x_unet = F.leaky_relu(self.unet_fc(x_unet))
#         x = F.relu(self.fc1(x_unet))
#         x = self.fc2(x)
#         return x
