import torch
import torch.nn as nn
import torch.nn.functional as F

# Updated Self-Attention Module remains the same
class SelfAttention3D(nn.Module):
    def __init__(self, in_dim, use_gamma=False):
        super(SelfAttention3D, self).__init__()
        self.in_dim = in_dim
        self.use_gamma = use_gamma
        # First 1x1x1 convolution (Wf)
        self.Wf = nn.Conv3d(in_dim, in_dim, kernel_size=1)
        # Second 1x1x1 convolution (Wφ)
        self.Wphi = nn.Conv3d(in_dim, in_dim, kernel_size=1)
        # Wv layer
        self.Wv = nn.Conv3d(in_dim, in_dim, kernel_size=1)
        # Sigmoid activation
        self.sigmoid = nn.Sigmoid()
        # Optional gamma parameter
        if self.use_gamma:
            self.gamma = nn.Parameter(torch.zeros(1))
        # Softmax over spatial dimensions
        self.softmax = nn.Softmax(dim=2)  # N = D*H*W

    def forward(self, x):
        B, C, D, H, W = x.size()
        N = D * H * W

        # Reshape x to (B, C, N)
        x_flat = x.view(B, C, N)

        # Compute f(x) = Wf x
        f_x = self.Wf(x).view(B, C, N)

        # Compute attention weights η_j
        eta = self.softmax(f_x)  # Shape: (B, C, N)

        # Compute φ(x) = Wφ x
        phi_x = self.Wphi(x).view(B, C, N)

        # Compute weighted sum ∑ η_j ⋅ φ(x_j)
        weighted_phi = eta * phi_x  # Element-wise multiplication
        summed_phi = torch.sum(weighted_phi, dim=2, keepdim=True)  # Sum over N

        # Apply Wv: v(x) = Wv x
        v = self.Wv(summed_phi.view(B, C, 1, 1, 1))

        # Apply sigmoid activation
        attention_map = self.sigmoid(v)

        # Optionally scale attention map with gamma
        if self.use_gamma:
            attention_map = self.gamma * attention_map

        # Expand attention map to match input dimensions
        attention_map = attention_map.expand_as(x)

        # Apply attention map to input
        out = attention_map * x

        return out

# Updated Pyramid Convolution Block without channel reduction
class PyramidConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes):
        super(PyramidConvBlock, self).__init__()
        self.paths = nn.ModuleList()
        for kernel_size in kernel_sizes:
            padding = kernel_size // 2
            path = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
                nn.ReLU(inplace=True),
                nn.Conv3d(out_channels, out_channels, kernel_size=kernel_size, padding=padding),
                nn.ReLU(inplace=True)
            )
            self.paths.append(path)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        outputs = []
        for path in self.paths:
            outputs.append(path(x))
        # Concatenate along the channel dimension
        out = torch.cat(outputs, dim=1)
        out = self.relu(out)
        return out

# Adjusted Generator Network to accommodate increased channels
class Generator(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_features=32):
        super(Generator, self).__init__()
        # Contracting path
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

        # First PyramidConvBlock with kernel sizes [3,5,7]
        self.down1 = PyramidConvBlock(in_channels, base_features, kernel_sizes=[3,5,7])
        features1 = base_features * len([3,5,7])

        # Second PyramidConvBlock with kernel sizes [3,5]
        self.down2 = PyramidConvBlock(features1, base_features*2, kernel_sizes=[3,5])
        features2 = base_features*2 * len([3,5])

        # Third Convolution Block
        self.down3 = nn.Sequential(
            nn.Conv3d(features2, base_features*4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(base_features*4, base_features*4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        features3 = base_features*4

        # Expanding path
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True),
            nn.Conv3d(features3, base_features*2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(base_features*2, base_features*2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        features_up1 = base_features*2

        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True),
            nn.Conv3d(features_up1, base_features, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(base_features, base_features, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        features_up2 = base_features

        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True),
            nn.Conv3d(features_up2, base_features, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(base_features, base_features, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        features_up3 = base_features

        # Self-Attention Module
        self.attention = SelfAttention3D(features_up3)

        # Final output layer
        self.final_conv = nn.Conv3d(features_up3, out_channels, kernel_size=1)

    def forward(self, x):
        # Contracting path
        x1 = self.down1(x)
        p1 = self.pool(x1)

        x2 = self.down2(p1)
        p2 = self.pool(x2)

        x3 = self.down3(p2)
        # Expanding path
        x = self.up1(x3)
        x = self.up2(x)
        x = self.up3(x)
        # Apply self-attention after the last up-convolutional block
        x = self.attention(x)
        # Final Convolution
        x = self.final_conv(x)
        return x

# Standard Discriminator remains unchanged
class StandardDiscriminator(nn.Module):
    def __init__(self, in_channels=1, features=[32, 64, 128, 256, 512]):
        super(StandardDiscriminator, self).__init__()
        layers = []
        prev_channels = in_channels
        for feature in features:
            layers.append(
                nn.Conv3d(prev_channels, feature, kernel_size=3, stride=2, padding=1)
            )
            layers.append(nn.BatchNorm3d(feature))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            prev_channels = feature
        self.model = nn.Sequential(*layers)
        self.classifier = nn.Conv3d(prev_channels, 1, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.model(x)
        x = self.classifier(x)
        x = torch.sigmoid(x)
        return x

# Task-Induced Discriminator remains unchanged
class _DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super(_DenseLayer, self).__init__()
        self.bn = nn.BatchNorm3d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv3d(in_channels, growth_rate, kernel_size=3, padding=1)

    def forward(self, x):
        new_features = self.conv(self.relu(self.bn(x)))
        return torch.cat([x, new_features], 1)

class _Transition(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(_Transition, self).__init__()
        self.bn = nn.BatchNorm3d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        self.pool = nn.AvgPool3d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(self.relu(self.bn(x)))
        x = self.pool(x)
        return x

class TaskInducedDiscriminator(nn.Module):
    def __init__(self, in_channels=1, num_classes=2, growth_rate=32, block_config=(6,12,24,16)):
        super(TaskInducedDiscriminator, self).__init__()
        num_init_features = 64
        # Initial Convolution and Pooling
        self.features = nn.Sequential(
            nn.Conv3d(in_channels, num_init_features, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm3d(num_init_features),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        )
        # Dense Blocks
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = self._make_dense_block(_DenseLayer, num_features, growth_rate, num_layers)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_features, num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2
        # Final Batch Norm
        self.features.add_module('norm5', nn.BatchNorm3d(num_features))
        # Classifier
        self.classifier = nn.Linear(num_features, num_classes)

    def _make_dense_block(self, block, in_channels, growth_rate, num_layers):
        layers = []
        for i in range(num_layers):
            layers.append(block(in_channels + i * growth_rate, growth_rate))
        return nn.Sequential(*layers)

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool3d(out, (1,1,1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        out = F.softmax(out, dim=1)
        return out

# Loss functions remain unchanged
def L1_loss(pred, target):
    return F.l1_loss(pred, target)

def ssim_loss(pred, target):
    # Assuming pred and target are normalized between 0 and 1
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu_x = F.avg_pool3d(pred, kernel_size=3, stride=1, padding=1)
    mu_y = F.avg_pool3d(target, kernel_size=3, stride=1, padding=1)

    sigma_x = F.avg_pool3d(pred ** 2, kernel_size=3, stride=1, padding=1) - mu_x ** 2
    sigma_y = F.avg_pool3d(target ** 2, kernel_size=3, stride=1, padding=1) - mu_y ** 2
    sigma_xy = F.avg_pool3d(pred * target, kernel_size=3, stride=1, padding=1) - mu_x * mu_y

    ssim_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    ssim_d = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)

    ssim = ssim_n / (ssim_d + 1e-8)
    loss = torch.clamp((1 - ssim) / 2, 0, 1)
    return loss.mean()

def combined_loss(G, Dstd, Dtask, real_MRI, real_PET, labels, gamma=1.0, lambda_=1.0, zeta=1.0):
    # Generate fake PET from MRI
    fake_PET = G(real_MRI)

    # Standard Discriminator Loss
    Dstd_real = Dstd(real_PET)
    Dstd_fake = Dstd(fake_PET.detach())

    # Labels for real and fake images
    real_labels = torch.ones_like(Dstd_real)
    fake_labels = torch.zeros_like(Dstd_fake)

    # Standard Discriminator Loss
    Dstd_loss_real = F.binary_cross_entropy(Dstd_real, real_labels)
    Dstd_loss_fake = F.binary_cross_entropy(Dstd_fake, fake_labels)
    LDstd = Dstd_loss_real + Dstd_loss_fake

    # Generator Loss
    Dstd_fake_for_G = Dstd(fake_PET)
    LG = F.binary_cross_entropy(Dstd_fake_for_G, real_labels)

    # Task-induced discriminator loss
    Dtask_output = Dtask(fake_PET)
    LDtask = F.cross_entropy(Dtask_output, labels)

    # Pixel-wise losses
    L1 = L1_loss(fake_PET, real_PET)
    LSSIM = ssim_loss(fake_PET, real_PET)

    # Combined Loss
    L = gamma * (L1 + LSSIM) + lambda_ * LG + zeta * LDtask
    return L, {'L1': L1.item(), 'LSSIM': LSSIM.item(), 'LG': LG.item(), 'LDstd': LDstd.item(), 'LDtask': LDtask.item()}

# Example usage:
if __name__ == "__main__":
    # Create random tensors to simulate MRI and PET images
    real_MRI = torch.randn((1, 1, 64, 64, 64))  # Batch size of 1, single channel, 64x64x64 volume
    real_PET = torch.randn((1, 1, 64, 64, 64))
    labels = torch.tensor([1])  # Assuming binary classification (AD vs Normal)

    # Instantiate models
    G = Generator()
    Dstd = StandardDiscriminator()
    Dtask = TaskInducedDiscriminator(num_classes=2)

    # Optimizers
    optimizer_G = torch.optim.Adam(G.parameters(), lr=1e-4)
    optimizer_Dstd = torch.optim.Adam(Dstd.parameters(), lr=1e-4)
    optimizer_Dtask = torch.optim.Adam(Dtask.parameters(), lr=1e-4)

    # Training loop example
    for epoch in range(1):  # Replace with actual number of epochs
        # Zero the parameter gradients
        optimizer_G.zero_grad()
        optimizer_Dstd.zero_grad()
        optimizer_Dtask.zero_grad()

        # Forward pass
        fake_PET = G(real_MRI)

        # Compute losses
        L, loss_dict = combined_loss(G, Dstd, Dtask, real_MRI, real_PET, labels)

        # Backward pass and optimization
        # Update Generator
        L.backward()
        optimizer_G.step()

        # Update Standard Discriminator
        Dstd_loss = loss_dict['LDstd']
        Dstd_loss = torch.tensor(Dstd_loss, requires_grad=True)
        Dstd_loss.backward()
        optimizer_Dstd.step()

        # Update Task-Induced Discriminator
        LDtask = loss_dict['LDtask']
        LDtask = torch.tensor(LDtask, requires_grad=True)
        LDtask.backward()
        optimizer_Dtask.step()

        print(f"Epoch [{epoch+1}], Losses: {loss_dict}")
