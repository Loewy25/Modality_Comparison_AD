import os
import numpy as np
import nibabel as nib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.functional import interpolate
from torchvision import models
from sklearn.model_selection import train_test_split
from scipy.stats import zscore
from scipy.ndimage import zoom

# Import data loading functions (Ensure these are correctly implemented)
from data_loading import generate_data_path_less, generate, binarylabel

# ------------------------------------------------------------
# Custom Dataset Class for Data Loading
# ------------------------------------------------------------
class CustomDataset(Dataset):
    def __init__(self, mri_images, pet_images):
        self.mri_images = mri_images
        self.pet_images = pet_images

    def __len__(self):
        return len(self.mri_images)

    def __getitem__(self, idx):
        mri = self.mri_images[idx]
        pet = self.pet_images[idx]
        return torch.FloatTensor(mri), torch.FloatTensor(pet)

# ------------------------------------------------------------
# Corrected DenseUNetGenerator Class
# ------------------------------------------------------------
class DenseUNetGenerator(nn.Module):
    def __init__(self, input_channels=1, output_channels=1, num_layers_per_block=2):
        super(DenseUNetGenerator, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.num_layers_per_block = num_layers_per_block

        # Initial convolution block
        self.initial_conv = self.convolution_block(self.input_channels, 64)

        # Downsampling path
        self.down_dense_blocks = nn.ModuleList()
        self.down_transition_layers = nn.ModuleList()
        self.filters_list = [64, 128, 256, 512]

        in_channels = 64
        skip_channels = []
        for filters in self.filters_list:
            # Dense Block
            dense_block = self.dense_block(in_channels, filters, num_layers_per_block)
            self.down_dense_blocks.append(dense_block)
            # Update in_channels after dense block
            in_channels = in_channels + num_layers_per_block * filters
            skip_channels.append(in_channels)
            # Transition Layer
            transition = self.transition_layer(in_channels, filters)
            self.down_transition_layers.append(transition)
            in_channels = filters  # Reset in_channels to filters after transition

        # Bottleneck dense block
        self.bottleneck_dense_block = self.dense_block(in_channels, in_channels, num_layers_per_block)
        in_channels = in_channels + num_layers_per_block * in_channels

        # Upsampling path
        self.up_upsampling_blocks = nn.ModuleList()
        self.up_dense_blocks = nn.ModuleList()
        for idx, filters in enumerate(reversed(self.filters_list)):
            skip_in_channels = skip_channels[-(idx+1)]
            # Upsampling block
            up_block = self.upsampling_block(in_channels, filters)
            self.up_upsampling_blocks.append(up_block)
            # After concatenation, in_channels becomes filters + skip_in_channels
            in_channels = filters + skip_in_channels
            # Dense Block
            dense_block = self.dense_block(in_channels, filters, num_layers_per_block)
            self.up_dense_blocks.append(dense_block)
            # Update in_channels after dense block
            in_channels = in_channels + num_layers_per_block * filters

        # Final Convolution
        self.final_conv = nn.Conv3d(in_channels, self.output_channels, kernel_size=1)
        self.activation = nn.Tanh()

    def convolution_block(self, in_channels, out_channels, kernel_size=3, stride=1):
        layers = [
            nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=kernel_size//2),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        return nn.Sequential(*layers)

    def dense_block(self, in_channels, growth_rate, num_layers=2):
        layers = nn.ModuleList()
        for i in range(num_layers):
            layers.append(self.single_dense_layer(in_channels + i * growth_rate, growth_rate))
        return layers

    def single_dense_layer(self, in_channels, growth_rate):
        layers = nn.Sequential(
            nn.Conv3d(in_channels, growth_rate, kernel_size=3, padding=1),
            nn.InstanceNorm3d(growth_rate),
            nn.LeakyReLU(0.2, inplace=True)
        )
        return layers

    def transition_layer(self, in_channels, out_channels):
        layers = [
            nn.Conv3d(in_channels, out_channels, kernel_size=1, padding=0),
            nn.InstanceNorm3d(out_channels),
            nn.AvgPool3d(kernel_size=2, stride=2)
        ]
        return nn.Sequential(*layers)

    def upsampling_block(self, in_channels, out_channels):
        layers = [
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        return nn.Sequential(*layers)

    def forward(self, x):
        skips = []
        x = self.initial_conv(x)

        # Downsampling Path
        for dense_block, transition in zip(self.down_dense_blocks, self.down_transition_layers):
            # Dense Block
            for layer in dense_block:
                out = layer(x)
                x = torch.cat([x, out], dim=1)
            skips.append(x)
            # Transition Layer
            x = transition(x)

        # Bottleneck Dense Block
        for layer in self.bottleneck_dense_block:
            out = layer(x)
            x = torch.cat([x, out], dim=1)

        # Upsampling Path
        for idx, (up_block, dense_block) in enumerate(zip(self.up_upsampling_blocks, self.up_dense_blocks)):
            skip = skips.pop()
            x = up_block(x)
            x = torch.cat([x, skip], dim=1)
            for layer in dense_block:
                out = layer(x)
                x = torch.cat([x, out], dim=1)

        # Final Convolution
        x = self.final_conv(x)
        x = self.activation(x)
        return x

# ------------------------------------------------------------
# ResNetEncoder Class with KL-Divergence Constraint
# ------------------------------------------------------------
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=False):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.downsample = downsample
        if downsample or stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm3d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x) if self.downsample else x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out

class ResNetEncoder(nn.Module):
    def __init__(self, input_channels=1, latent_dim=512):
        super(ResNetEncoder, self).__init__()
        self.input_channels = input_channels
        self.latent_dim = latent_dim

        self.conv1 = nn.Conv3d(self.input_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        # ResNet layers
        self.layer1 = self._make_layer(64, 64, blocks=3)
        self.layer2 = self._make_layer(64, 128, blocks=4, stride=2)
        self.layer3 = self._make_layer(128, 256, blocks=6, stride=2)
        self.layer4 = self._make_layer(256, 512, blocks=3, stride=2)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc_mean = nn.Linear(512, self.latent_dim)
        self.fc_logvar = nn.Linear(512, self.latent_dim)

    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = []
        downsample = (stride != 1) or (in_channels != out_channels)
        layers.append(ResBlock(in_channels, out_channels, stride, downsample))
        for _ in range(1, blocks):
            layers.append(ResBlock(out_channels, out_channels))
        return nn.Sequential(*layers)


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Pass through ResNet layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        z_mean = self.fc_mean(x)
        z_log_var = self.fc_logvar(x)
        return z_mean, z_log_var

# ------------------------------------------------------------
# Discriminator Class with Patch-Level Discrimination
# ------------------------------------------------------------
class Discriminator(nn.Module):
    def __init__(self, input_channels=1):
        super(Discriminator, self).__init__()
        # Convolutional blocks without pooling
        self.conv_block1 = self.convolution_block(input_channels, 32, use_pool=False)
        self.conv_block2 = self.convolution_block(32, 64, use_pool=True)
        self.conv_block3 = self.convolution_block(64, 128, use_pool=True)
        self.conv_block4 = self.convolution_block(128, 256, use_pool=False)
        self.final_conv = nn.Conv3d(256, 1, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def convolution_block(self, in_channels, out_channels, use_pool):
        layers = [
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        if use_pool:
            layers.append(nn.MaxPool3d(kernel_size=2, stride=2))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv_block1(x)  # No pooling
        x = self.conv_block2(x)  # Pooling
        x = self.conv_block3(x)  # Pooling
        x = self.conv_block4(x)  # No pooling
        x = self.final_conv(x)
        x = self.sigmoid(x)
        return x


# ------------------------------------------------------------
# BMGAN Class with Integrated Loss Functions
# ------------------------------------------------------------
class BMGAN:
    def __init__(self, generator, discriminator, encoder, lambda1=10.0, lambda2=0.5):
        self.generator = generator.to(device)
        self.discriminator = discriminator.to(device)
        self.encoder = encoder.to(device)
        self.lambda1 = lambda1  # Weight for L1 Loss
        self.lambda2 = lambda2  # Weight for Perceptual Loss

        self.vgg_model = self.get_vgg_model().to(device)

        # Define optimizers
        self.optimizer_G = optim.Adam(self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.optimizer_D = optim.Adam(self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.optimizer_E = optim.Adam(self.encoder.parameters(), lr=0.0002, betas=(0.5, 0.999))

        # Define loss functions
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()

    def get_vgg_model(self):
        # Load the VGG16 model
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        # Extract features from an intermediate layer
        model = nn.Sequential(*list(vgg.features.children())[:9])  # Up to 'block2_pool'
        for param in model.parameters():
            param.requires_grad = False
        return model

    def perceptual_loss(self, y_true, y_pred):
        batch_size, channels, depth, height, width = y_true.size()
        y_true = y_true.permute(0, 2, 1, 3, 4).contiguous().view(batch_size * depth, channels, height, width)
        y_pred = y_pred.permute(0, 2, 1, 3, 4).contiguous().view(batch_size * depth, channels, height, width)

        # Resize to VGG16 input size (224x224)
        y_true = interpolate(y_true, size=(224, 224), mode='bilinear', align_corners=False)
        y_pred = interpolate(y_pred, size=(224, 224), mode='bilinear', align_corners=False)

        # Convert grayscale to RGB by repeating channels
        y_true = y_true.repeat(1, 3, 1, 1)
        y_pred = y_pred.repeat(1, 3, 1, 1)

        # Pass through VGG model to extract features
        y_true_features = self.vgg_model(y_true)
        y_pred_features = self.vgg_model(y_pred)

        # Compute the mean absolute error between the features
        perceptual_loss = self.l1_loss(y_pred_features, y_true_features)

        return perceptual_loss

    def l1_perceptual_loss(self, y_true, y_pred):
        l1_loss = self.l1_loss(y_pred, y_true)
        perceptual_loss = self.perceptual_loss(y_true, y_pred)
        return l1_loss + self.lambda2 * perceptual_loss

    def kl_divergence_loss(self, z_mean, z_log_var):
        kl_loss = -0.5 * torch.mean(1 + z_log_var - z_mean.pow(2) - z_log_var.exp())
        return kl_loss

    def lsgan_loss(self, y_pred, y_true):
        return self.mse_loss(y_pred, y_true)

    def train(self, mri_images, pet_images, epochs, batch_size):
        # Split data into training and validation sets (80% training, 20% validation)
        mri_train, mri_val, pet_train, pet_val = train_test_split(mri_images, pet_images, test_size=0.2, random_state=42)
        
        # Create DataLoaders for training and validation
        train_dataset = CustomDataset(mri_train, pet_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        val_dataset = CustomDataset(mri_val, pet_val)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        real_label = 1.0
        fake_label = 0.0
    
        for epoch in range(epochs):
            # Set models to training mode
            self.generator.train()
            self.discriminator.train()
            self.encoder.train()
    
            total_d_loss = 0
            total_g_loss = 0
    
            # Training Loop
            for i, (real_mri, real_pet) in enumerate(train_loader):
                real_mri = real_mri.to(device)
                real_pet = real_pet.to(device)
                batch_size = real_mri.size(0)
    
                # ---------------------
                #  Train Discriminator
                # ---------------------
                self.discriminator.zero_grad()
    
                # For real images
                output_real = self.discriminator(real_pet)
                label_real = torch.ones_like(output_real, device=real_pet.device)
                d_loss_real = self.lsgan_loss(output_real, label_real)
                
                # For fake images
                fake_pet = self.generator(real_mri)
                output_fake = self.discriminator(fake_pet.detach())
                label_fake = torch.zeros_like(output_fake, device=real_pet.device)
                d_loss_fake = self.lsgan_loss(output_fake, label_fake)
                
                # Total discriminator loss
                d_loss = (d_loss_real + d_loss_fake) * 0.5

                self.optimizer_D.step()
    
                # -----------------
                #  Train Generator and Encoder
                # -----------------
                self.generator.zero_grad()
                self.encoder.zero_grad()
    
                # GAN loss
                output_fake = self.discriminator(fake_pet)
                label_real = torch.ones_like(output_fake, device=real_pet.device)
                g_gan_loss = self.lsgan_loss(output_fake, label_real)
    
                # L1 and Perceptual loss
                l1_perc_loss = self.l1_perceptual_loss(real_pet, fake_pet)
    
                # KL divergence loss
                z_mean_real, z_log_var_real = self.encoder(real_pet)
                z_mean_fake, z_log_var_fake = self.encoder(fake_pet)
                kl_loss_real = self.kl_divergence_loss(z_mean_real, z_log_var_real)
                kl_loss_fake = self.kl_divergence_loss(z_mean_fake, z_log_var_fake)
                kl_loss = kl_loss_real + kl_loss_fake
    
                # Total Generator loss
                g_loss = g_gan_loss + self.lambda1 * l1_perc_loss + self.lambda2 * kl_loss
                g_loss.backward()
                self.optimizer_G.step()
                self.optimizer_E.step()
    
                # Accumulate losses for printing
                total_d_loss += d_loss.item()
                total_g_loss += g_loss.item()
    
            # Average losses per epoch
            avg_d_loss = total_d_loss / len(train_loader)
            avg_g_loss = total_g_loss / len(train_loader)
            
            print(f"Epoch [{epoch+1}/{epochs}] Training D Loss: {avg_d_loss:.4f}, G Loss: {avg_g_loss:.4f}")
    
            # ---------------------------
            # Validation Step (no gradients)
            # ---------------------------
            self.generator.eval()  # Set generator to evaluation mode
            self.discriminator.eval()
            validation_loss = 0
            with torch.no_grad():  # No gradient calculation for validation
                for real_mri, real_pet in val_loader:
                    real_mri = real_mri.to(device)
                    real_pet = real_pet.to(device)
                    
                    # Generate fake PET images
                    fake_pet = self.generator(real_mri)
                    
                    # Calculate validation loss (L1 Perceptual Loss)
                    val_loss = self.l1_perceptual_loss(real_pet, fake_pet)
                    validation_loss += val_loss.item()
    
            # Average validation loss per epoch
            validation_loss /= len(val_loader)
            print(f"Epoch [{epoch+1}/{epochs}] Validation Loss: {validation_loss:.4f}")


# ------------------------------------------------------------
# Data Loading and Preprocessing Functions
# ------------------------------------------------------------
def load_mri_pet_data(task):
    """
    Load and preprocess MRI and PET data.
    Args:
        task (str): Task identifier.
    Returns:
        Tuple of numpy arrays: (mri_data, pet_data)
    """
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

def resize_image(image, target_shape):
    """
    Resize a 3D image to the target shape using zoom.
    Args:
        image (numpy.ndarray): 3D image.
        target_shape (tuple): Desired shape.
    Returns:
        numpy.ndarray: Resized image.
    """
    zoom_factors = [target_shape[i] / image.shape[i] for i in range(3)]
    return zoom(image, zoom_factors, order=1)

# ------------------------------------------------------------
# Utility Function to Save Images
# ------------------------------------------------------------
def save_images(image, file_path):
    """
    Save a 3D image as a NIfTI file.
    Args:
        image (numpy.ndarray): 3D image.
        file_path (str): Destination file path.
    """
    nib.save(nib.Nifti1Image(image, np.eye(4)), file_path)

# ------------------------------------------------------------
# Main Function
# ------------------------------------------------------------
if __name__ == '__main__':

    # Check for available GPUs
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print("Using GPU:", torch.cuda.get_device_name(0))
    else:
        device = torch.device('cpu')
        print("Using CPU")

    # Define task and experiment info
    task = 'cd'
    info = 'experiment4'  # New parameter for the subfolder

    # Load MRI and PET data
    print("Loading MRI and PET data...")
    mri_data, pet_data = load_mri_pet_data(task)
    print(f"Loaded {mri_data.shape[0]} MRI and PET image pairs.")

    # Split data into training (2/3) and test (1/3)
    print("Splitting data into training and testing sets...")
    mri_train, mri_gen, pet_train, pet_gen = train_test_split(
        mri_data, pet_data, test_size=0.33, random_state=42
    )
    print(f"Training set: {mri_train.shape[0]} samples")
    print(f"Testing set: {mri_gen.shape[0]} samples")

    # Initialize generator
    print("Initializing Generator")
    generator = DenseUNetGenerator(input_channels=1, output_channels=1)
    generator.to(device)
    print(generator)

    # Initialize discriminator
    print("Initializing Discriminator")
    discriminator = Discriminator(input_channels=1)
    discriminator.to(device)
    print(discriminator)

    # Initialize encoder
    print("Initializing Encoder")
    encoder = ResNetEncoder(input_channels=1, latent_dim=512)
    encoder.to(device)
    print(encoder)

    # Initialize BMGAN model
    print("Building BMGAN model...")
    bmgan = BMGAN(generator, discriminator, encoder)

    # Train the model
    print("Starting training...")
    bmgan.train(mri_train, pet_train, epochs=500, batch_size=1)

    # Create directories to store the results
    output_dir_mri = f'gan/{task}/{info}/mri'
    output_dir_pet = f'gan/{task}/{info}/pet'

    os.makedirs(output_dir_mri, exist_ok=True)
    os.makedirs(output_dir_pet, exist_ok=True)
    print(f"Created directories: {output_dir_mri}, {output_dir_pet}")

    # Predict PET images for the test MRI data
    print("Generating PET images for the test set...")
    generator.eval()
    with torch.no_grad():
        generated_pet_images = []
        for mri in mri_gen:
            mri_tensor = torch.FloatTensor(mri).unsqueeze(0).to(device)
            generated_pet = generator(mri_tensor)
            generated_pet = generated_pet.cpu().numpy()
            generated_pet_images.append(generated_pet[0])

    # Save the test MRI data and the generated PET images in their respective folders
    print("Saving generated PET images and corresponding MRI scans...")
    for i in range(len(mri_gen)):
        mri_file_path = os.path.join(output_dir_mri, f'mri_{i}.nii.gz')
        pet_file_path = os.path.join(output_dir_pet, f'generated_pet_{i}.nii.gz')

        # Save MRI and generated PET images
        save_images(mri_gen[i][0], mri_file_path)  # Save MRI
        save_images(generated_pet_images[i][0], pet_file_path)  # Save generated PET

    # Print confirmation
    print(f"Saved {len(mri_gen)} MRI and corresponding generated PET images in 'gan/{task}/{info}'")



