import torch
import torch.nn as nn


class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_dim, hidden_dim, kernel_size=3, stride=1, padding=1)
        )







def main():
    # make a model
    # make a loss a dataset
    # make a dataloader
    # tain the model with evaluation on the validation set



if __name__ == "__main__":
    main()