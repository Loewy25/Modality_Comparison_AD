import torch
import torch.nn as nn
import torch.nn.functional as F
from data_loading import binarylabel

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

        f_x = self.Wf(x).view(B, C, N)
        phi_x = self.Wphi(x).view(B, C, N)

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

        # Down 1
        self.down1 = PyramidConvBlock(in_channels, 64, kernel_sizes=[3, 5])
        features1 = 64 * 2  # 128

        # Down 2
        self.down2 = PyramidConvBlock(features1, 128, kernel_sizes=[3, 5])
        features2 = 128 * 2  # 256

        # Down 3
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
        self.up1 = nn.Sequential(
            nn.ConvTranspose3d(features3, features3, kernel_size=2, stride=2),
            nn.ReLU(inplace=True)
        )
        self.dec_conv1 = nn.Sequential(
            nn.Conv3d(features3 + features3, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        # Up 2:
        self.up2 = nn.Sequential(
            nn.ConvTranspose3d(256, 256, kernel_size=2, stride=2),
            nn.ReLU(inplace=True)
        )
        self.dec_conv2 = nn.Sequential(
            nn.Conv3d(256 + features2, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        # Up 3:
        self.up3 = nn.Sequential(
            nn.ConvTranspose3d(128, 128, kernel_size=2, stride=2),
            nn.ReLU(inplace=True)
        )
        self.dec_conv3 = nn.Sequential(
            nn.Conv3d(128 + features1, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        # Attention and final conv
        self.attention = SelfAttention3D(64)  # 64 channels in
        self.conv = nn.Conv3d(64, 64, kernel_size=3, padding=1)
        self.final_conv = nn.Conv3d(64, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
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
        x = self.classifier(x)  # shape [B, 1, D', H', W']
        x = F.adaptive_avg_pool3d(x, (1,1,1))  # now [B, 1, 1, 1, 1]
        x = x.view(x.size(0), -1)  # now [B, 1]
        x = torch.sigmoid(x)
        return x


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
    def __init__(self, in_channels, growth_rate=16):
        super(_DenseBlock, self).__init__()
        self.layer1 = _DenseLayer(in_channels, growth_rate)
        self.layer2 = _DenseLayer(in_channels + growth_rate, growth_rate)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x


class _Transition(nn.Module):
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
        
        self.init_conv = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2)
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

        # Dense Block 4
        self.db4 = _DenseBlock(num_features, growth_rate)
        num_features = num_features + 2 * growth_rate

        self.final_conv = nn.Sequential(
            nn.Conv3d(num_features, num_features, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2)
        )

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

        x = self.final_conv(x)

        x = F.adaptive_avg_pool3d(x, (1, 1, 1))
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = F.softmax(x, dim=1)
        return x

def L1_loss(pred, target):
    return F.l1_loss(pred, target)

def ssim_loss(pred, target):
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

import os
import numpy as np
import nibabel as nib
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from scipy.stats import zscore
from scipy.ndimage import zoom
from math import log10
import matplotlib.pyplot as plt
from data_loading import generate_data_path_less, generate

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

    label = binarylabel(label, task)

    mri_resized = []
    pet_resized = []
    for mri_path, pet_path in zip(mri_data, pet_data):
        mri_img = nib.load(mri_path).get_fdata()
        pet_img = nib.load(pet_path).get_fdata()

        mri_img = zscore(mri_img, axis=None)
        pet_img = zscore(pet_img, axis=None)

        mri_resized.append(resize_image(mri_img, (128, 128, 128)))
        pet_resized.append(resize_image(pet_img, (128, 128, 128)))

    mri_resized = np.expand_dims(np.array(mri_resized), 1)
    pet_resized = np.expand_dims(np.array(pet_resized), 1)
    label = np.array(label, dtype=np.int64)

    return mri_resized, pet_resized, label

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_step(G, Dstd, Dtask, real_MRI, real_PET, labels,
               optimizer_G, optimizer_Dstd, optimizer_Dtask,
               gamma=1.0, lambda_=1.0, zeta=1.0, device='cpu'):
    batch_size = real_MRI.size(0)
    real_labels = torch.ones((batch_size, 1), device=device)
    fake_labels = torch.zeros((batch_size, 1), device=device)

    fake_PET = G(real_MRI)

    optimizer_Dstd.zero_grad()
    Dstd_real = Dstd(real_PET)
    Dstd_fake = Dstd(fake_PET.detach())

    Dstd_loss_real = nn.functional.binary_cross_entropy(Dstd_real, real_labels)
    Dstd_loss_fake = nn.functional.binary_cross_entropy(Dstd_fake, fake_labels)
    LDstd = Dstd_loss_real + Dstd_loss_fake
    LDstd.backward()
    optimizer_Dstd.step()

    optimizer_Dtask.zero_grad()
    Dtask_output = Dtask(fake_PET.detach())
    LDtask = nn.functional.cross_entropy(Dtask_output, labels)
    LDtask.backward()
    optimizer_Dtask.step()

    optimizer_G.zero_grad()
    Dstd_fake_for_G = Dstd(fake_PET)
    LG = nn.functional.binary_cross_entropy(Dstd_fake_for_G, real_labels)

    Dtask_output_for_G = Dtask(fake_PET)
    LDtask_for_G = nn.functional.cross_entropy(Dtask_output_for_G, labels)

    L_1 = L1_loss(fake_PET, real_PET)
    L_SSIM = ssim_loss(fake_PET, real_PET)

    L_G = gamma * (L_1 + L_SSIM) + lambda_ * LG + zeta * LDtask_for_G
    L_G.backward()
    optimizer_G.step()

    loss_dict = {
        'L1': L_1.item(),
        'LSSIM': L_SSIM.item(),
        'LG': LG.item(),
        'LDstd': LDstd.item(),
        'LDtask': LDtask.item(),
        'LDtask_for_G': LDtask_for_G.item()
    }

    return loss_dict


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    task = 'cd'
    info = 'Pyramid_2st_180_0.3_0.5/1.0/0.5_4-5'

    print("Loading MRI and PET data...")
    mri_data, pet_data, all_labels = load_mri_pet_data(task)
    print(f"Total samples: {mri_data.shape[0]}")

    mri_train, mri_test, pet_train, pet_test, labels_train, labels_test = train_test_split(
        mri_data, pet_data, all_labels, test_size=0.3, random_state=8
    )
    print(f"Train set: {mri_train.shape[0]} samples, Test set: {mri_test.shape[0]} samples")

    mri_train, mri_val, pet_train, pet_val, labels_train, labels_val = train_test_split(
        mri_train, pet_train, labels_train, test_size=0.2, random_state=42
    )

    G = Generator().to(device)
    Dstd = StandardDiscriminator().to(device)
    Dtask = TaskInducedDiscriminator(num_classes=2).to(device)

    print("Generator params:", count_parameters(G))
    print("StandardDiscriminator params:", count_parameters(Dstd))
    print("TaskInducedDiscriminator params:", count_parameters(Dtask))

    output_dir_mri = f'gan/{task}/{info}/mri'
    output_dir_pet = f'gan/{task}/{info}/pet'
    output_dir_real_pet = f'gan/{task}/{info}/real_pet'
    output_dir = f'gan/{task}/{info}'

    os.makedirs(output_dir_mri, exist_ok=True)
    os.makedirs(output_dir_pet, exist_ok=True)
    os.makedirs(output_dir_real_pet, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

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
            label = self.labels[idx]
            return torch.FloatTensor(mri), torch.FloatTensor(pet), label

    train_dataset = CustomDataset(mri_train, pet_train, labels_train)
    val_dataset = CustomDataset(mri_val, pet_val, labels_val)
    test_dataset = CustomDataset(mri_test, pet_test, labels_test)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

    optimizer_G = optim.Adam(G.parameters(), lr=1e-4)
    optimizer_Dstd = optim.Adam(Dstd.parameters(), lr=4e-5)
    optimizer_Dtask = optim.Adam(Dtask.parameters(), lr=1e-4)

    gamma = 0.5
    lambda_ = 1.0
    zeta = 0.5

    best_val_loss = float('inf')
    patience = 500
    epochs_no_improve = 0
    best_G_state = None

    training_generator_losses = []
    training_discriminator_losses = []
    validation_losses = []

    epochs = 180
    for epoch in range(epochs):
        G.train()
        Dstd.train()
        Dtask.train()

        total_g_loss = 0
        total_d_loss = 0

        for real_mri, real_pet, labels in train_loader:
            real_mri = real_mri.to(device)
            real_pet = real_pet.to(device)
            labels = torch.LongTensor(labels).to(device)

            loss_dict = train_step(G, Dstd, Dtask, real_mri, real_pet, labels,
                                   optimizer_G, optimizer_Dstd, optimizer_Dtask,
                                   gamma=gamma, lambda_=lambda_, zeta=zeta, device=device)
            total_g_loss += loss_dict['LG']
            total_d_loss += loss_dict['LDstd']

        avg_g_loss = total_g_loss / len(train_loader)
        avg_d_loss = total_d_loss / len(train_loader)
        training_generator_losses.append(avg_g_loss)
        training_discriminator_losses.append(avg_d_loss)

        G.eval()
        with torch.no_grad():
            val_loss = 0
            total_mae = 0
            total_psnr = 0
            num_val_batches = 0

            for real_mri, real_pet, labels in val_loader:
                real_mri = real_mri.to(device)
                real_pet = real_pet.to(device)
                labels = torch.LongTensor(labels).to(device)

                fake_pet = G(real_mri)
                l1_val = L1_loss(fake_pet, real_pet)
                ssim_val = ssim_loss(fake_pet, real_pet)
                this_val_loss = (l1_val + ssim_val).item()
                val_loss += this_val_loss

                mae = compute_mae(real_pet, fake_pet)
                psnr = compute_psnr(real_pet, fake_pet)
                total_mae += mae
                total_psnr += psnr
                num_val_batches += 1

            val_loss /= num_val_batches
            validation_losses.append(val_loss)
            avg_mae = total_mae / num_val_batches
            avg_psnr = total_psnr / num_val_batches

        print(f"Epoch [{epoch+1}/{epochs}] Training G Loss: {avg_g_loss:.4f}, D Loss: {avg_d_loss:.4f}, Val Loss: {val_loss:.4f}, MAE: {avg_mae:.4f}, PSNR: {avg_psnr:.2f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_G_state = G.state_dict()
        else:
            epochs_no_improve += 1
            if epochs_no_improve > patience:
                print("Early stopping triggered.")
                break

    if best_G_state is not None:
        G.load_state_dict(best_G_state)
        print("Loaded best model weights based on validation loss.")

    plt.figure()
    plt.plot(training_generator_losses, label='Generator Training Loss')
    plt.plot(training_discriminator_losses, label='Discriminator Training Loss')
    plt.plot(validation_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Curves')
    plt.savefig(os.path.join(output_dir, 'loss_curve.png'))
    plt.close()

    G.eval()
    total_mae = 0
    total_psnr = 0
    num_test_batches = 0
    with torch.no_grad():
        for real_mri, real_pet, labels in test_loader:
            real_mri = real_mri.to(device)
            real_pet = real_pet.to(device)
            labels = torch.LongTensor(labels).to(device)

            fake_pet = G(real_mri)
            mae = compute_mae(real_pet, fake_pet)
            psnr = compute_psnr(real_pet, fake_pet)
            total_mae += mae
            total_psnr += psnr
            num_test_batches += 1

    avg_mae = total_mae / num_test_batches
    avg_psnr = total_psnr / num_test_batches
    print(f"\nTest MAE: {avg_mae:.4f}, Test PSNR: {avg_psnr:.2f}")

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
