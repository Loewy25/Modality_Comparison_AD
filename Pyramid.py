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
        self.final_conv = nn.Conv3d(64, out_channels, kernel_size=3, padding=1)


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
import os
import numpy as np
import nibabel as nib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from scipy.stats import zscore
from scipy.ndimage import zoom
from math import log10
import torchvision.transforms as transforms
from torchvision.utils import save_image
import tempfile
from pytorch_fid import fid_score
import matplotlib.pyplot as plt

# Assuming generate_data_path_less, generate, binarylabel are defined elsewhere
from data_loading import generate_data_path_less, generate


def L1_loss(pred, target):
    return nn.functional.l1_loss(pred, target)

def ssim_loss(pred, target):
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu_x = nn.functional.avg_pool3d(pred, kernel_size=3, stride=1, padding=1)
    mu_y = nn.functional.avg_pool3d(target, kernel_size=3, stride=1, padding=1)

    sigma_x = nn.functional.avg_pool3d(pred ** 2, kernel_size=3, stride=1, padding=1) - mu_x ** 2
    sigma_y = nn.functional.avg_pool3d(target ** 2, kernel_size=3, stride=1, padding=1) - mu_y ** 2
    sigma_xy = nn.functional.avg_pool3d(pred * target, kernel_size=3, stride=1, padding=1) - mu_x * mu_y

    ssim_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    ssim_d = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)

    ssim = ssim_n / (ssim_d + 1e-8)
    loss = torch.clamp((1 - ssim) / 2, 0, 1)
    return loss.mean()

def compute_mae(real_pet, fake_pet):
    return torch.mean(torch.abs(real_pet - fake_pet)).item()

def compute_psnr(real_pet, fake_pet):
    mse = nn.functional.mse_loss(real_pet, fake_pet).item()
    if mse == 0:
        return float('inf')
    max_I = 20.0
    psnr = 10 * log10((max_I ** 2) / mse)
    return psnr

def save_images_nii(image, file_path):
    nib.save(nib.Nifti1Image(image, np.eye(4)), file_path)

def resize_image(image, target_shape):
    zoom_factors = [target_shape[i] / image.shape[i] for i in range(3)]
    return zoom(image, zoom_factors, order=1)

def load_mri_pet_data(task):
    images_pet, images_mri, labels = generate_data_path_less()
    pet_data, label = generate(images_pet, labels, task)
    mri_data, label = generate(images_mri, labels, task)

    mri_resized = []
    pet_resized = []
    for mri_path, pet_path in zip(mri_data, pet_data):
        mri_img = nib.load(mri_path).get_fdata()
        pet_img = nib.load(pet_path).get_fdata()
        mri_img = zscore(mri_img, axis=None)
        pet_img = zscore(pet_img, axis=None)
        mri_resized.append(resize_image(mri_img, (128, 128, 128)))
        pet_resized.append(resize_image(pet_img, (128, 128, 128)))

    return np.expand_dims(np.array(mri_resized), 1), np.expand_dims(np.array(pet_resized), 1)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Training step
import os
import numpy as np
import nibabel as nib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from math import log10
import torchvision.transforms as transforms
from torchvision.utils import save_image
import tempfile
from pytorch_fid import fid_score
import matplotlib.pyplot as plt
from scipy.stats import zscore
from scipy.ndimage import zoom

# ---------------------
# Assuming these functions and classes are defined:
#  - load_mri_pet_data(task)
#  - generate_data_path_less(), generate() [referenced in code, not shown]
#  - CustomDataset that now returns (mri, pet, label)
#  - L1_loss, ssim_loss, compute_mae, compute_psnr, save_images_nii defined
#  - Generator, StandardDiscriminator, TaskInducedDiscriminator defined as above
# ---------------------

# Modify CustomDataset to also provide labels
class CustomDataset(Dataset):
    def __init__(self, mri_images, pet_images, labels):
        self.mri_images = mri_images
        self.pet_images = pet_images
        self.labels = labels

    def __len__(self):
        return len(self.mri_images)

    def __getitem__(self, idx):
        mri = self.mri_images[idx]
        pet = self.pet_images[idx]
        label = self.labels[idx]  # Ensure these are integers for classification
        return torch.FloatTensor(mri), torch.FloatTensor(pet), torch.tensor(label, dtype=torch.long)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_step_gan(G, Dstd, real_MRI, real_PET, optimizer_G, optimizer_Dstd, gamma=1.0, lambda_=0.5):
    # Train G + Dstd only (no Dtask)
    batch_size = real_MRI.size(0)
    device = real_MRI.device
    real_labels = torch.ones((batch_size, 1), device=device)
    fake_labels = torch.zeros((batch_size, 1), device=device)

    # Generate Fake PET
    fake_PET = G(real_MRI)

    # Update Dstd
    optimizer_Dstd.zero_grad()
    Dstd_real = Dstd(real_PET)
    Dstd_fake = Dstd(fake_PET.detach())

    Dstd_loss_real = nn.functional.binary_cross_entropy(Dstd_real, real_labels)
    Dstd_loss_fake = nn.functional.binary_cross_entropy(Dstd_fake, fake_labels)
    LDstd = Dstd_loss_real + Dstd_loss_fake
    LDstd.backward()
    optimizer_Dstd.step()

    # Update G
    optimizer_G.zero_grad()
    Dstd_fake_for_G = Dstd(fake_PET)
    LG = nn.functional.binary_cross_entropy(Dstd_fake_for_G, real_labels)

    L_1 = L1_loss(fake_PET, real_PET)
    L_SSIM = ssim_loss(fake_PET, real_PET)
    L_G = gamma * (L_1 + L_SSIM) + lambda_ * LG  # No Dtask here
    L_G.backward()
    optimizer_G.step()

    return L_G.item(), LDstd.item()

def pretrain_gan(G, Dstd, train_loader, val_loader, device, epochs=50, gamma=1.0, lambda_=0.5):
    optimizer_G = optim.Adam(G.parameters(), lr=1e-4)
    optimizer_Dstd = optim.Adam(Dstd.parameters(), lr=4e-4)

    best_val_loss = float('inf')
    best_G_state = None

    for epoch in range(epochs):
        G.train()
        Dstd.train()
        total_g_loss = 0
        total_d_loss = 0

        for real_mri, real_pet, labels in train_loader:
            real_mri = real_mri.to(device)
            real_pet = real_pet.to(device)
            g_loss, d_loss = train_step_gan(G, Dstd, real_mri, real_pet, optimizer_G, optimizer_Dstd, gamma=gamma, lambda_=lambda_)
            total_g_loss += g_loss
            total_d_loss += d_loss

        # Validation (just L1+SSIM)
        G.eval()
        val_loss = 0
        num_val_batches = 0
        with torch.no_grad():
            for real_mri, real_pet, labels in val_loader:
                real_mri = real_mri.to(device)
                real_pet = real_pet.to(device)
                fake_pet = G(real_mri)
                l1_val = L1_loss(fake_pet, real_pet)
                ssim_val = ssim_loss(fake_pet, real_pet)
                val_loss += (l1_val + ssim_val).item()
                num_val_batches += 1

        val_loss /= num_val_batches
        print(f"Pre-train GAN Epoch [{epoch+1}/{epochs}]: G Loss: {total_g_loss/len(train_loader):.4f}, Dstd Loss: {total_d_loss/len(train_loader):.4f}, Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_G_state = G.state_dict()

    if best_G_state is not None:
        G.load_state_dict(best_G_state)
        print("Loaded best pre-trained GAN weights.")

def pretrain_dtask(Dtask, train_loader, val_loader, device, epochs=50):
    # Train Dtask on real PET + labels
    optimizer_Dtask = optim.Adam(Dtask.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_Dtask_state = None

    for epoch in range(epochs):
        Dtask.train()
        correct = 0
        total = 0
        for real_mri, real_pet, labels in train_loader:
            real_pet = real_pet.to(device)
            labels = labels.to(device)

            optimizer_Dtask.zero_grad()
            out = Dtask(real_pet)
            loss = criterion(out, labels)
            loss.backward()
            optimizer_Dtask.step()

            _, predicted = torch.max(out, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
        train_acc = correct / total

        # Val
        Dtask.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for real_mri, real_pet, labels in val_loader:
                real_pet = real_pet.to(device)
                labels = labels.to(device)
                out = Dtask(real_pet)
                _, predicted = torch.max(out, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
        val_acc = correct / total

        print(f"Pre-train Dtask Epoch [{epoch+1}/{epochs}]: Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_Dtask_state = Dtask.state_dict()

    if best_Dtask_state is not None:
        Dtask.load_state_dict(best_Dtask_state)
        print("Loaded best pre-trained Dtask weights.")

def fine_tune_tpa_gan(G, Dstd, Dtask, train_loader, val_loader, device, epochs=50, gamma=1.0, lambda_=0.5, zeta=0.5):
    # Freeze Dtask
    for param in Dtask.parameters():
        param.requires_grad = False

    optimizer_G = optim.Adam(G.parameters(), lr=1e-4)
    optimizer_Dstd = optim.Adam(Dstd.parameters(), lr=4e-4)

    criterionCE = nn.CrossEntropyLoss()
    best_val_loss = float('inf')
    best_G_state = None

    for epoch in range(epochs):
        G.train()
        Dstd.train()
        Dtask.eval()  # Frozen

        total_g_loss = 0
        total_d_loss = 0

        for real_mri, real_pet, labels in train_loader:
            real_mri = real_mri.to(device)
            real_pet = real_pet.to(device)
            labels = labels.to(device)

            # Generate Fake PET
            fake_PET = G(real_mri)

            # Update Dstd
            optimizer_Dstd.zero_grad()
            Dstd_real = Dstd(real_pet)
            Dstd_fake = Dstd(fake_PET.detach())
            real_labels = torch.ones_like(Dstd_real)
            fake_labels = torch.zeros_like(Dstd_fake)
            Dstd_loss_real = nn.functional.binary_cross_entropy(Dstd_real, real_labels)
            Dstd_loss_fake = nn.functional.binary_cross_entropy(Dstd_fake, fake_labels)
            LDstd = Dstd_loss_real + Dstd_loss_fake
            LDstd.backward()
            optimizer_Dstd.step()

            # Update G
            optimizer_G.zero_grad()
            Dstd_fake_for_G = Dstd(fake_PET)
            LG = nn.functional.binary_cross_entropy(Dstd_fake_for_G, real_labels)

            Dtask_output_for_G = Dtask(fake_PET)
            LDtask_for_G = criterionCE(Dtask_output_for_G, labels)

            L_1 = L1_loss(fake_PET, real_pet)
            L_SSIM = ssim_loss(fake_PET, real_pet)

            L_G = gamma * (L_1 + L_SSIM) + lambda_ * LG + zeta * LDtask_for_G
            L_G.backward()
            optimizer_G.step()

            total_g_loss += LG.item()
            total_d_loss += LDstd.item()

        # Validation
        G.eval()
        with torch.no_grad():
            val_loss = 0
            num_val_batches = 0
            for real_mri, real_pet, labels in val_loader:
                real_mri = real_mri.to(device)
                real_pet = real_pet.to(device)
                fake_pet = G(real_mri)
                l1_val = L1_loss(fake_pet, real_pet)
                ssim_val = ssim_loss(fake_pet, real_pet)
                val_loss += (l1_val + ssim_val).item()
                num_val_batches += 1

            val_loss /= num_val_batches

        print(f"Fine-tune Epoch [{epoch+1}/{epochs}]: G Loss: {total_g_loss/len(train_loader):.4f}, Dstd Loss: {total_d_loss/len(train_loader):.4f}, Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_G_state = G.state_dict()

    if best_G_state is not None:
        G.load_state_dict(best_G_state)
        print("Loaded best fine-tuned TPA-GAN weights.")

##############################
# Main script starts here
##############################

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    task = 'cd'
    info = 'my_tpa_gan_run'

    # Load data (must return arrays and labels)
    # Ensure that load_mri_pet_data(task) returns mri_data, pet_data
    mri_data, pet_data = load_mri_pet_data(task)
    # You must have labels_data aligned with these MRI/PET pairs.
    # labels_data must be a numpy array of same length as mri_data/pet_data, with class labels.
    # Example (dummy):
    labels_data = np.ones(len(mri_data), dtype=int)  # Replace with real labels

    # Split
    # Adjust test_size and val splits as desired.
    mri_train, mri_test, pet_train, pet_test, labels_train, labels_test = train_test_split(
        mri_data, pet_data, labels_data, test_size=0.15, random_state=8)
    mri_train, mri_val, pet_train, pet_val, labels_train, labels_val = train_test_split(
        mri_train, pet_train, labels_train, test_size=0.2, random_state=42)

    # Initialize models
    G = Generator().to(device)
    Dstd = StandardDiscriminator().to(device)
    Dtask = TaskInducedDiscriminator(num_classes=2).to(device)

    print("Generator params:", count_parameters(G))
    print("StandardDiscriminator params:", count_parameters(Dstd))
    print("TaskInducedDiscriminator params:", count_parameters(Dtask))

    output_dir = f'gan/{task}/{info}'
    os.makedirs(output_dir, exist_ok=True)

    # Dataloaders with batch size = 2 as per description
    BATCH_SIZE = 2
    train_dataset = CustomDataset(mri_train, pet_train, labels_train)
    val_dataset = CustomDataset(mri_val, pet_val, labels_val)
    test_dataset = CustomDataset(mri_test, pet_test, labels_test)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Phase 1: Pre-train GAN (G+Dstd)
    pretrain_gan(G, Dstd, train_loader, val_loader, device, epochs=50)

    # Phase 2: Pre-train Dtask
    pretrain_dtask(Dtask, train_loader, val_loader, device, epochs=50)

    # Phase 3: Fine-tune combined model
    fine_tune_tpa_gan(G, Dstd, Dtask, train_loader, val_loader, device, epochs=50)

    # Evaluation on test set
    G.eval()
    total_mae = 0
    total_psnr = 0
    num_test_batches = 0
    real_images_list = []
    fake_images_list = []
    with torch.no_grad():
        for real_mri, real_pet, labels in test_loader:
            real_mri = real_mri.to(device)
            real_pet = real_pet.to(device)
            fake_pet = G(real_mri)
            mae = compute_mae(real_pet, fake_pet)
            psnr = compute_psnr(real_pet, fake_pet)
            total_mae += mae
            total_psnr += psnr
            num_test_batches += 1
            real_images_list.append(real_pet.cpu())
            fake_images_list.append(fake_pet.cpu())

    avg_mae = total_mae / num_test_batches
    avg_psnr = total_psnr / num_test_batches
    print(f"\nTest MAE: {avg_mae:.4f}, Test PSNR: {avg_psnr:.2f}")

    # (Optional) Compute FID on test set
    real_test_dir = os.path.join(tempfile.gettempdir(), 'real_images_test')
    fake_test_dir = os.path.join(tempfile.gettempdir(), 'fake_images_test')
    os.makedirs(real_test_dir, exist_ok=True)
    os.makedirs(fake_test_dir, exist_ok=True)

    def save_for_fid(images_list, directory):
        os.makedirs(directory, exist_ok=True)
        for idx, img_tensor in enumerate(images_list):
            for b_idx in range(img_tensor.size(0)):
                vol = img_tensor[b_idx,0].numpy()
                slice_img = torch.from_numpy(vol[vol.shape[0]//2])
                slice_img = slice_img.unsqueeze(0)
                save_image(slice_img, os.path.join(directory, f"{idx}_{b_idx}.png"))

    save_for_fid(real_images_list, real_test_dir)
    save_for_fid(fake_images_list, fake_test_dir)
    fid_value = fid_score.calculate_fid_given_paths([real_test_dir, fake_test_dir], batch_size=4, device=device, dims=2048)
    print(f"Test FID: {fid_value:.4f}")

    # Save final outputs if desired
    output_dir_mri = f'gan/{task}/{info}/mri'
    output_dir_pet = f'gan/{task}/{info}/pet'
    output_dir_real_pet = f'gan/{task}/{info}/real_pet'

    os.makedirs(output_dir_mri, exist_ok=True)
    os.makedirs(output_dir_pet, exist_ok=True)
    os.makedirs(output_dir_real_pet, exist_ok=True)

    G.eval()
    generated_pet_images = []
    with torch.no_grad():
        for i in range(len(mri_test)):
            mri_tensor = torch.FloatTensor(mri_test[i]).unsqueeze(0).to(device)
            fake_pet = G(mri_tensor)
            fake_pet = fake_pet.cpu().numpy()
            generated_pet_images.append(fake_pet[0])

    for i in range(len(mri_test)):
        mri_file_path = os.path.join(output_dir_mri, f'mri_{i}.nii.gz')
        pet_file_path = os.path.join(output_dir_pet, f'generated_pet_{i}.nii.gz')
        real_pet_file_path = os.path.join(output_dir_real_pet, f'real_pet_{i}.nii.gz')

        save_images_nii(mri_test[i][0], mri_file_path)
        save_images_nii(generated_pet_images[i][0], pet_file_path)
        save_images_nii(pet_test[i][0], real_pet_file_path)

    print(f"Saved {len(mri_test)} MRI scans, generated PET images, and real PET images in 'gan/{task}/{info}'")
