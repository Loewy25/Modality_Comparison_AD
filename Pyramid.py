import torch
import torch.nn as nn
import torch.nn.functional as F

# Self-Attention Module
class SelfAttention3D(nn.Module):
    def __init__(self, in_dim):
        super(SelfAttention3D, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv3d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv3d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv3d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps (B x C x D x H x W)
            returns :
                out : self attention value + input feature
                attention: B x N x N (N=D*H*W)
        """
        m_batchsize, C, depth, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, depth*height*width).permute(0, 2, 1)  # B x N x C
        proj_key = self.key_conv(x).view(m_batchsize, -1, depth*height*width)  # B x C x N
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # B x N x N
        proj_value = self.value_conv(x).view(m_batchsize, -1, depth*height*width)  # B x C x N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))  # B x C x N
        out = out.view(m_batchsize, C, depth, height, width)

        out = self.gamma * out + x
        return out

# Pyramid Convolution Block
class PyramidConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes):
        super(PyramidConvBlock, self).__init__()
        self.convs = nn.ModuleList()
        for kernel_size in kernel_sizes:
            padding = kernel_size // 2
            self.convs.append(
                nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        outputs = []
        for conv in self.convs:
            outputs.append(conv(x))
        out = sum(outputs) / len(outputs)
        out = self.relu(out)
        return out

# Generator Network
class Generator(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=32):
        super(Generator, self).__init__()
        # Contracting path
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.down1 = PyramidConvBlock(in_channels, features, kernel_sizes=[3,5,7])
        self.down2 = PyramidConvBlock(features, features*2, kernel_sizes=[3,5])
        self.down3 = nn.Sequential(
            nn.Conv3d(features*2, features*4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        # Expanding path
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True),
            nn.Conv3d(features*4, features*2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(features*2, features*2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True),
            nn.Conv3d(features*2, features, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(features, features, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True),
            nn.Conv3d(features, features, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(features, features, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        # Self-Attention Module
        self.attention = SelfAttention3D(features)
        # Final output layer
        self.final_conv = nn.Conv3d(features, out_channels, kernel_size=1)

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
        # Self-Attention
        x = self.attention(x)
        # Final Convolution
        x = self.final_conv(x)
        return x

# Standard Discriminator
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

# Dense Block for Task-Induced Discriminator
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

# Task-Induced Discriminator
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

# L1 Loss
def L1_loss(pred, target):
    return F.l1_loss(pred, target)

# SSIM Loss
# For SSIM Loss, we need to implement it or use an existing implementation.
# Here, we'll use a simple approximation.
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

    ssim = ssim_n / ssim_d
    loss = torch.clamp((1 - ssim) / 2, 0, 1)
    return loss.mean()

# Combined Loss Function
def combined_loss(G, Dstd, Dtask, real_MRI, real_PET, labels, gamma=1.0, lambda_=1.0, zeta=1.0):
    # Generate fake PET from MRI
    fake_PET = G(real_MRI)
    # GAN Losses
    Dstd_real = Dstd(real_PET)
    Dstd_fake = Dstd(fake_PET.detach())

    LG = torch.mean(torch.log(1 - Dstd_fake + 1e-8))
    LDstd = torch.mean(torch.log(1 - Dstd_real + 1e-8) + torch.log(Dstd_fake + 1e-8))

    # Task-induced discriminator loss
    Dtask_output = Dtask(fake_PET)
    LDtask = F.cross_entropy(Dtask_output, labels)

    # Pixel-wise losses
    L1 = L1_loss(fake_PET, real_PET)
    LSSIM = ssim_loss(fake_PET, real_PET)

    # Combined Loss
    L = gamma * (L1 + LSSIM) + lambda_ * (LG + LDstd) + zeta * LDtask
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

    # Forward pass
    fake_PET = G(real_MRI)

    # Compute losses
    L, loss_dict = combined_loss(G, Dstd, Dtask, real_MRI, real_PET, labels)

    # Backward pass and optimization
    optimizer_G.zero_grad()
    L.backward()
    optimizer_G.step()

    optimizer_Dstd.zero_grad()
    optimizer_Dstd.step()

    optimizer_Dtask.zero_grad()
    optimizer_Dtask.step()

    print(f"Losses: {loss_dict}")
