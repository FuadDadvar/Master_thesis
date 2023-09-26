import torch
from VAE_ import VAE  # Replace with your actual import

# Initialize your model with the required dimensions
input_dim = 64  # replace with actual input dimension
hidden_dim = 128  # replace with actual hidden dimension
latent_dim = 64  # replace with actual latent dimension

model = VAE(input_dim=input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)

# Create a dummy input tensor with the same shape as your actual input.
# For this example, I'm assuming the input is a 2D tensor, you might need to adjust
# it according to your actual input shape.
# Here, 16 is the batch size and input_dim is the number of features in your input.
dummy_input = torch.randn(16, input_dim)

# Pass the dummy input through the model
try:
    model.eval()
    with torch.no_grad():
        reconstructed_batch, mu, log_var = model(dummy_input)
    print('\n\n')
    print("Dry run successful! The network works with the given input.")
    print('\n\n')
except Exception as e:
    print('\n\n')
    print(f"Dry run failed! Error: {e}")
    print('\n\n')