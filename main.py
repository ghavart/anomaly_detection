import torch
import torch.nn as nn
import argparse
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter

from autoencoder_model import Autoencoder
from dataset import ImageDataset

SAVE_ROOT = Path("./checkpoints")

def main(args):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # make a model
    model = Autoencoder(bottlencek_dim=128)

    # make a loss function
    loss_func = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4, weight_decay=1e-5)

    batch_size = 32
    workers = 4

    # make a dataloader
    train_dataset = ImageDataset(args.data_dir) 
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=workers,
                                               pin_memory=True)
    # tain the model with evaluation on the validation set

    # make a tensorboard writer
    tsb_writer = SummaryWriter('./tensorboard')

    epochs = 10
    save_interval = 2

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for i, batch in enumerate(train_loader):
            target_batch = batch.to(device=device)
            recon_batch = model(target_batch)
            loss = loss_func(recon_batch, target_batch)

            # backprop step 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            del target_batch, recon_batch, loss
            torch.cuda.empty_cache()

        tsb_writer.add_scalar('Loss/train', running_loss, epoch)
        print(f'Epoch {epoch} loss: {running_loss}')

        if epoch % save_interval == 0:
            torch.save(model.state_dict(), f'{SAVE_ROOT}/model_{epoch}.pth')


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--data_dir', type=str, default='/mnt/data/coco/images/train2017')
    args = argparser.parse_args() 
    main(args)