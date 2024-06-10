
import torch.utils.data as utils


def train(X, y, model, criterion, optimiser, epochs = 500):
    losses = []

    for epoch in range(epochs):
        out  = model(X)
        loss = criterion(out, y)

        losses.append(loss.item())

        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        if epoch % 10 == 9:
            print(f"Epoch {epoch + 1:>3} - MSE: {loss.item()}")

    return losses


def train_batched(X, y, model, criterion, optimiser, epochs = 500, batch_size = 20, shuffle = False):
    losses = []

    dataset    = utils.TensorDataset(X, y)
    dataloader = utils.DataLoader(dataset, batch_size = batch_size, shuffle = shuffle)

    for epoch in range(epochs):
        for batch, (X_batch, y_batch) in enumerate(dataloader):
            out  = model(X_batch)
            loss = criterion(out, y_batch)

            losses.append(loss.item())

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            if batch % 10 == 9:
                print(f"Epoch {epoch + 1:>3} - Batch {batch + 1:>3} - MSE: {loss.item()}")

    return losses
