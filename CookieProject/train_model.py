import click
import torch
from models.model import MyAwesomeModel
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

from data.make_dataset import mnist


@click.group()
def cli():
    """Command line interface."""
    pass


@click.command()
@click.option("--lr", default=1e-3, help="learning rate to use for training")
def train(lr):
    """Train a model on MNIST."""
    print("Training day and night")
    print(lr)

    # TODO: Implement training loop here
    model = MyAwesomeModel()
    train_set, _ = mnist()

    criterion = torch.nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    epochs = 50
    train_losses = []

    for e in tqdm(range(epochs)):
        model.train()
        running_loss = 0.0
        for images, labels in train_set:
            optimizer.zero_grad()

            lgps = model(images)
            loss = criterion(lgps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_losses.append(running_loss / len(train_set))

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    torch.save(model.state_dict(), "../models/trained_model_{timestamp}.pt")

    # Save the training curve plot
    plt.plot(train_losses)
    plt.xlabel("Training Steps")
    plt.ylabel("Training Loss")
    plt.title("Training Loss Curve")
    plt.savefig("../reports/figures/training_loss_curve_{timestamp}.png")


@click.command()
@click.argument("model_checkpoint")
def evaluate(model_checkpoint):
    """Evaluate a trained model."""
    print("Evaluating like my life dependends on it")
    print(model_checkpoint)

    # TODO: Implement evaluation logic here
    model = MyAwesomeModel()
    model.load_state_dict(torch.load(model_checkpoint))
    _, test_set = mnist()
    criterion = torch.nn.NLLLoss()

    test_loss = 0
    accuracy = 0

    with torch.no_grad():
        model.eval()
        for images, labels in tqdm(test_set):
            lgps = model(images)
            test_loss += criterion(lgps, labels)

            ps = torch.exp(lgps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor))

    print(f"Accuracy: {accuracy.item()/len(test_set)*100:.3f}%")


cli.add_command(train)
cli.add_command(evaluate)


if __name__ == "__main__":
    cli()
