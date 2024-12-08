import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention3D(nn.Module):
    def __init__(self, in_dim):
        super(SelfAttention3D, self).__init__()
        self.in_dim = in_dim
        self.Wf = nn.Conv3d(in_dim, in_dim, kernel_size=1)
        self.Wphi = nn.Conv3d(in_dim, in_dim, kernel_size=1)
        self.Wv = nn.Conv3d(in_dim, in_dim, kernel_size=1)
        self.softmax = nn.Softmax(dim=2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        B, C, D, H, W = x.size()
        N = D * H * W

        f_x = self.Wf(x).view(B, C, N)    # f(x)
        phi_x = self.Wphi(x).view(B, C, N)# φ(x)

        # Compute attention weights
        eta = self.softmax(f_x)  # (B, C, N)

        # Weighted sum over φ(x)
        weighted_phi = eta * phi_x
        summed_phi = torch.sum(weighted_phi, dim=2, keepdim=True) # (B, C, 1)

        v = self.Wv(summed_phi.view(B, C, 1, 1, 1))  # (B, C, 1, 1, 1)
        attention_map = self.sigmoid(v) # Produces a mask

        return attention_map


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
        outputs = [path(x) for path in self.paths]
        out = torch.cat(outputs, dim=1)
        out = self.relu(out)
        return out


class BottleneckBlock(nn.Module):
    def __init__(self, in_channels, num_blocks):
        super(BottleneckBlock, self).__init__()
        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            block = nn.Sequential(
                nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            )
            self.blocks.append(block)

    def forward(self, x):
        for block in self.blocks:
            residual = x
            out = block(x)
            x = residual + out
        return x


class Generator(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_features=32):
        super(Generator, self).__init__()
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

        # Down 1: want 128 channels total
        # Use 2 paths so each path out_channels=64 => total 128
        self.down1 = PyramidConvBlock(in_channels, 64, kernel_sizes=[3, 5])
        features1 = 64 * 2  # 128

        # Down 2: want 256 channels total
        # Again 2 paths, each 128 => total 256
        self.down2 = PyramidConvBlock(features1, 128, kernel_sizes=[3, 5])
        features2 = 128 * 2  # 256

        # Down 3: want 512 channels total
        self.down3 = nn.Sequential(
            nn.Conv3d(features2, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        features3 = 512

        # Bottleneck
        self.bottleneck_conv1 = nn.Conv3d(features3, features3, kernel_size=3, padding=1)
        self.bottleneck_relu1 = nn.ReLU(inplace=True)
        self.bottleneck_conv2 = nn.Conv3d(features3, features3, kernel_size=3, padding=1)
        self.bottleneck_relu2 = nn.ReLU(inplace=True)
        self.bottleneck = BottleneckBlock(features3, num_blocks=6) # stays at 512

        # Up 1:
        # Transpose conv: from 512 -> 256 (to match features2)
        self.up1 = nn.Sequential(
            nn.ConvTranspose3d(features3, features3, kernel_size=2, stride=2),
            nn.ReLU(inplace=True)
        )
        # After concat with x3: 512+512=1024 -> reduce to 256
        self.dec_conv1 = nn.Sequential(
            nn.Conv3d(features3 + features3, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        # Up 2:
        # From 256 -> 256 using transpose (no channel change here, just spatial up)
        # Actually, we need to feed dec_conv1 output (256 ch) to up2:
        # Let’s keep up2 from 256 -> 256
        self.up2 = nn.Sequential(
            nn.ConvTranspose3d(256, 256, kernel_size=2, stride=2),
            nn.ReLU(inplace=True)
        )
        # After concat with x2: 256+256=512 -> reduce to 128
        self.dec_conv2 = nn.Sequential(
            nn.Conv3d(256 + features2, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        # Up 3:
        # From 128 -> 128 using transpose
        self.up3 = nn.Sequential(
            nn.ConvTranspose3d(128, 128, kernel_size=2, stride=2),
            nn.ReLU(inplace=True)
        )
        # After concat with x1: 128+128=256 -> reduce to 64
        self.dec_conv3 = nn.Sequential(
            nn.Conv3d(128 + features1, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        # Attention and final conv
        self.attention = SelfAttention3D(64)  # 64 channels in
        self.conv = nn.Conv3d(64, 64, kernel_size=3, padding=1)
        self.final_conv = nn.Conv3d(64, out_channels, kernel_size=3)

    def forward(self, x):
        # Down
        x1 = self.down1(x)   # 128 ch
        p1 = self.pool(x1)

        x2 = self.down2(p1)  # 256 ch
        p2 = self.pool(x2)

        x3 = self.down3(p2)  # 512 ch
        p3 = self.pool(x3)

        # Bottleneck
        bottleneck = self.bottleneck_conv1(p3)
        bottleneck = self.bottleneck_relu1(bottleneck)
        bottleneck = self.bottleneck_conv2(bottleneck)
        bottleneck = self.bottleneck_relu2(bottleneck)
        bottleneck = self.bottleneck(bottleneck)  # 512 ch

        # Up 1
        x = self.up1(bottleneck)     # still 512 ch
        x = torch.cat([x, x3], dim=1)  # 512+512=1024
        x = self.dec_conv1(x)        # 1024->256 ch

        # Up 2
        x = self.up2(x)              # 256 ch
        x = torch.cat([x, x2], dim=1) # 256+256=512
        x = self.dec_conv2(x)        # 512->128 ch

        # Up 3
        x = self.up3(x)              # 128 ch
        x = torch.cat([x, x1], dim=1) # 128+128=256
        x = self.dec_conv3(x)        # 256->64 ch

        # Attention and conv
        att = self.attention(x)      # 64->64 (mask)
        conv = self.conv(x)          # 64->64
        x = att * conv               # still 64 ch

        x = self.final_conv(x)       # 64->1
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
import torch
import torch.nn as nn
import torch.nn.functional as F

class _DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate=16):
        super(_DenseLayer, self).__init__()
        self.bn = nn.BatchNorm3d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv3d(in_channels, growth_rate, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        out = self.conv(self.relu(self.bn(x)))
        return torch.cat([x, out], dim=1)


class _DenseBlock(nn.Module):
    # Each dense block has exactly 2 dense layers (cat ×2)
    def __init__(self, in_channels, growth_rate=16):
        super(_DenseBlock, self).__init__()
        self.layer1 = _DenseLayer(in_channels, growth_rate)
        self.layer2 = _DenseLayer(in_channels + growth_rate, growth_rate)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x


class _Transition(nn.Module):
    # Transition: 1x1 conv + AvgPool s2
    def __init__(self, in_channels, out_channels):
        super(_Transition, self).__init__()
        self.bn = nn.BatchNorm3d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False)
        self.pool = nn.AvgPool3d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(self.relu(self.bn(x)))
        x = self.pool(x)
        return x


class TaskInducedDiscriminator(nn.Module):
    def __init__(self, in_channels=1, num_classes=2, growth_rate=16):
        super(TaskInducedDiscriminator, self).__init__()
        
        # Initial: 3×3 conv + max pool s2 (as per figure: initial downsampling)
        self.init_conv = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2)  # max pool s2
        )

        num_features = 32

        # Dense Block 1
        self.db1 = _DenseBlock(num_features, growth_rate)
        num_features = num_features + 2 * growth_rate
        self.trans1 = _Transition(num_features, num_features // 2)
        num_features = num_features // 2

        # Dense Block 2
        self.db2 = _DenseBlock(num_features, growth_rate)
        num_features = num_features + 2 * growth_rate
        self.trans2 = _Transition(num_features, num_features // 2)
        num_features = num_features // 2

        # Dense Block 3
        self.db3 = _DenseBlock(num_features, growth_rate)
        num_features = num_features + 2 * growth_rate
        self.trans3 = _Transition(num_features, num_features // 2)
        num_features = num_features // 2

        # Dense Block 4 (no transition after last block)
        self.db4 = _DenseBlock(num_features, growth_rate)
        num_features = num_features + 2 * growth_rate

        # Final step: 3×3 conv + max pool s2 (instead of final bn)
        self.final_conv = nn.Sequential(
            nn.Conv3d(num_features, num_features, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2)
        )

        # Fully connected layer with softmax
        self.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        x = self.init_conv(x)

        x = self.db1(x)
        x = self.trans1(x)

        x = self.db2(x)
        x = self.trans2(x)

        x = self.db3(x)
        x = self.trans3(x)

        x = self.db4(x)

        # Apply the final conv+relu+maxpool block
        x = self.final_conv(x)

        # Global average pool to 1x1x1
        x = F.adaptive_avg_pool3d(x, (1, 1, 1))
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = F.softmax(x, dim=1)
        return x

# Example usage:
# model = TaskInducedDiscriminator(in_channels=1, num_classes=2)
# input = torch.randn(1, 1, 64, 64, 64) # Example input volume
# output = model(input)
# print(output.shape) # should be [1, 2]


import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------
# Loss Functions
# ---------------------
def L1_loss(pred, target):
    return F.l1_loss(pred, target)

def ssim_loss(pred, target):
    # pred and target assumed normalized [0,1] or similar
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

# ---------------------
# Example Training Step
# ---------------------
def train_step(G, Dstd, Dtask, real_MRI, real_PET, labels,
               optimizer_G, optimizer_Dstd, optimizer_Dtask,
               gamma=1.0, lambda_=1.0, zeta=1.0):
    # Switch to training mode
    G.train()
    Dstd.train()
    Dtask.train()

    batch_size = real_MRI.size(0)
    device = real_MRI.device

    # Generate Fake PET
    fake_PET = G(real_MRI)

    # ---------------------------------------
    # 1) Update Standard Discriminator (Dstd)
    # ---------------------------------------
    optimizer_Dstd.zero_grad()

    # Real PET → should be real (label=1)
    Dstd_real = Dstd(real_PET)
    real_labels = torch.ones_like(Dstd_real, device=device)

    # Fake PET → should be fake (label=0)
    Dstd_fake = Dstd(fake_PET.detach())
    fake_labels = torch.zeros_like(Dstd_fake, device=device)

    LDstd_real = F.binary_cross_entropy(Dstd_real, real_labels)
    LDstd_fake = F.binary_cross_entropy(Dstd_fake, fake_labels)
    LDstd = LDstd_real + LDstd_fake

    LDstd.backward()
    optimizer_Dstd.step()

    # ---------------------------------------
    # 2) Update Task-Induced Discriminator (Dtask)
    # ---------------------------------------
    optimizer_Dtask.zero_grad()

    # Dtask tries to classify the generated PET correctly
    # Assume labels: 0 for Normal, 1 for AD (just example)
    Dtask_output = Dtask(fake_PET.detach())
    LDtask = F.cross_entropy(Dtask_output, labels)

    LDtask.backward()
    optimizer_Dtask.step()

    # ---------------------------------------
    # 3) Update Generator (G)
    # ---------------------------------------
    optimizer_G.zero_grad()

    # Generator wants to fool Dstd → classify fake as real
    Dstd_fake_for_G = Dstd(fake_PET)
    LG = F.binary_cross_entropy(Dstd_fake_for_G, real_labels)

    # Generator also wants to produce PET that helps correct classification in Dtask
    Dtask_output_for_G = Dtask(fake_PET)  # no detach here
    # The generator's perspective: it wants Dtask to be confident in the true label
    # Minimizing cross-entropy wrt. G parameters encourages G to produce better PET
    LDtask_for_G = F.cross_entropy(Dtask_output_for_G, labels)

    # Pixel-wise losses (L1 + SSIM)
    L_1 = L1_loss(fake_PET, real_PET)
    L_SSIM = ssim_loss(fake_PET, real_PET)

    # Combined loss for Generator
    # L = gamma(L1+LSSIM) + lambda LG + zeta LDtask
    L_G = gamma * (L_1 + L_SSIM) + lambda_ * LG + zeta * LDtask_for_G

    L_G.backward()
    optimizer_G.step()

    # ---------------------------------------
    # Collect Losses for Logging
    # ---------------------------------------
    loss_dict = {
        'L1': L_1.item(),
        'LSSIM': L_SSIM.item(),
        'LG': LG.item(),
        'LDstd': LDstd.item(),
        'LDtask': LDtask.item(),
        'LDtask_for_G': LDtask_for_G.item()
    }

    return loss_dict

# ---------------------
# Example usage
# ---------------------
if __name__ == "__main__":
    # Assuming Generator, StandardDiscriminator, TaskInducedDiscriminator are defined
    # and match the architectures described in the paper and previous code examples.
    G = Generator()  # your TPA-GAN generator
    Dstd = StandardDiscriminator()
    Dtask = TaskInducedDiscriminator(num_classes=2)

    # Move to device if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    G.to(device)
    Dstd.to(device)
    Dtask.to(device)

    # Example data
    real_MRI = torch.randn((1, 1, 64, 64, 64), device=device)
    real_PET = torch.randn((1, 1, 64, 64, 64), device=device)
    labels = torch.tensor([1], device=device)  # e.g., 1 indicating AD

    # Optimizers
    optimizer_G = torch.optim.Adam(G.parameters(), lr=1e-4)
    optimizer_Dstd = torch.optim.Adam(Dstd.parameters(), lr=1e-4)
    optimizer_Dtask = torch.optim.Adam(Dtask.parameters(), lr=1e-4)

    # Training loop example
    for epoch in range(10):  # Just an example, replace with actual epochs
        loss_dict = train_step(G, Dstd, Dtask, real_MRI, real_PET, labels,
                               optimizer_G, optimizer_Dstd, optimizer_Dtask,
                               gamma=1.0, lambda_=1.0, zeta=1.0)
        print(f"Epoch [{epoch+1}] Losses: {loss_dict}")

