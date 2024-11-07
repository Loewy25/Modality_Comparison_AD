import torch

def test_cuda():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"CUDA is available. Using device: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("CUDA is not available. Using CPU.")

    # Create a tensor and move it to the device
    x = torch.tensor([1.0, 2.0, 3.0]).to(device)
    print(x)

if __name__ == "__main__":
    test_cuda()
