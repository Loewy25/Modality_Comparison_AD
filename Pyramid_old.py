import torch
import torch.nn as nn
import torch.nn.functional as F
from data_loading import binarylabel
import os
import numpy as np
import nibabel as nib
from sklearn.model_selection import train_test_split
from scipy.stats import zscore
from scipy.ndimage import zoom
from math import log10
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

# ---------------------
# Models (Generator, Discriminators), and Losses
# are defined exactly as in your code
# ---------------------

# ... (Your Generator, Discriminators, and loss definitions remain unchanged) ...

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

    # Convert string labels to binary integers using binarylabel function
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
    
    # label should now be a list of 0/1
    label = np.array(label, dtype=np.int64)

    return mri_resized, pet_resized, label


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Training step
def train_step(G, Dstd, Dtask, real_MRI, real_PET, labels,
               optimizer_G, optimizer_Dstd, optimizer_Dtask,
               gamma=1.0, lambda_=1.0, zeta=1.0, device='cpu'):
    batch_size = real_MRI.size(0)
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

    # Update Dtask
    optimizer_Dtask.zero_grad()
    Dtask_output = Dtask(fake_PET.detach())
    LDtask = nn.functional.cross_entropy(Dtask_output, labels)
    LDtask.backward()
    optimizer_Dtask.step()

    # Update G
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

    # Load data (now also returns labels)
    print("Loading MRI and PET data...")
    mri_data, pet_data, all_labels = load_mri_pet_data(task)
    print(f"Total samples: {mri_data.shape[0]}")

    # Split into train/test (and then train/val) with labels
    mri_train, mri_test, pet_train, pet_test, labels_train, labels_test = train_test_split(
        mri_data, pet_data, all_labels, test_size=0.3, random_state=8
    )
    print(f"Train set: {mri_train.shape[0]} samples, Test set: {mri_test.shape[0]} samples")

    mri_train, mri_val, pet_train, pet_val, labels_train, labels_val = train_test_split(
        mri_train, pet_train, labels_train, test_size=0.2, random_state=42
    )

    # Initialize models
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

    # Updated CustomDataset to include labels
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
            label = self.labels[idx]  # Ensure label is an integer or long
            return torch.FloatTensor(mri), torch.FloatTensor(pet), label

    # Create DataLoaders with labels
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

        # Validation
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

    # Plot losses
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

    # Evaluation on test set
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

    # Generate PET images for the test set and save them
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
