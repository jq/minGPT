from sklearn.datasets import make_circles
import pandas as pd
import matplotlib.pyplot as plt
from helper_functions import plot_predictions, plot_decision_boundary
from torch import nn
import torch

class CircleModelV0(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(in_features=2, out_features=10)
        self.layer_2 = nn.Linear(in_features=10, out_features=10)
        self.layer_3 = nn.Linear(in_features=10, out_features=1)
        # without relu, it can' t model the circle
        self.relu = nn.ReLU() # <- add in ReLU activation function
    def forward(self, x):
        # return self.layer_3(self.layer_2(self.layer_1(x)))
        return self.layer_3(self.relu(self.layer_2(self.relu(self.layer_1(x)))))

if __name__ == "__main__":
    n_samples = 1000

    # Create circles
    X, y = make_circles(n_samples, noise=0.03,random_state=42)
    # Make DataFrame of circle data X[:, 0]是X 取第一列,
    circles = pd.DataFrame({"X1": X[:, 0],
                            "X2": X[:, 1],
                            "label": y
                            })
    print(circles.head(10))
    print(circles.label.value_counts())
    plt.scatter(x=X[:, 0],
                y=X[:, 1],
                c=y,
                cmap=plt.cm.RdYlBu);
    plt.show()
    X = torch.from_numpy(X).type(torch.float)
    y = torch.from_numpy(y).type(torch.float)
    # Split data into train and test sets
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.2, # 20% test, 80% train
                                                        random_state=42) # make the random split reproducible


    # Make device agnostic code
    device = "cuda" if torch.cuda.is_available() else "cpu"


    # 4. Create an instance of the model and send it to target device
    model_0 = CircleModelV0().to(device)

    # Replicate CircleModelV0 with nn.Sequential
    # model_0 = nn.Sequential(
    #     nn.Linear(in_features=2, out_features=10),
    #     nn.Linear(in_features=10, out_features=10),
    #     nn.Linear(in_features=10, out_features=1)
    # ).to(device)

    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(params=model_0.parameters(), lr=0.1)
    # Calculate accuracy (a classification metric)
    def accuracy_fn(y_true, y_pred):
        correct = torch.eq(y_true, y_pred).sum().item()
        acc = (correct / len(y_pred)) * 100
        return acc

    untrained_preds = model_0(X_test.to(device))
    # Use sigmoid on model logits
    y_pred_probs = torch.sigmoid(untrained_preds)
    y_preds = torch.round(y_pred_probs)
    y_pred_labels = torch.round(torch.sigmoid(model_0(X_test.to(device))))

    # Check for equality
    print(torch.eq(y_preds.squeeze(), y_pred_labels.squeeze()))

    accuracy_fn(y_pred_labels.squeeze(), y_preds.squeeze())

    torch.manual_seed(42)

    # Set the number of epochs
    epochs = 1000

    # Put data to target device
    X_train, y_train = X_train.to(device), y_train.to(device)
    X_test, y_test = X_test.to(device), y_test.to(device)

    # Build training and evaluation loop
    for epoch in range(epochs):
        model_0.train()
        # 1. Forward pass (model outputs raw logits) # squeeze to remove extra `1` dimensions, this won't work unless model and data are on same device
        y_logits = model_0(X_train).squeeze()
        y_pred = torch.round(torch.sigmoid(y_logits)) # turn logits -> pred probs -> pred labls
        loss = loss_fn(y_logits, # Using nn.BCEWithLogitsLoss works with raw logits
                       y_train)
        acc = accuracy_fn(y_true=y_train,
                          y_pred=y_pred)
        # how loss pass to optimizer
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        ### Testing
        model_0.eval()
        with torch.inference_mode():
            # 1. Forward pass
            test_logits = model_0(X_test).squeeze()
            test_pred = torch.round(torch.sigmoid(test_logits))
            # 2. Caculate loss/accuracy
            test_loss = loss_fn(test_logits,
                                y_test)
            test_acc = accuracy_fn(y_true=y_test,
                                   y_pred=test_pred)

        # Print out what's happening every 10 epochs
        if epoch % 10 == 0:
            print(f"Epoch: {epoch} | Loss: {loss:.5f}, Accuracy: {acc:.2f}% | Test loss: {test_loss:.5f}, Test acc: {test_acc:.2f}%")

    # Plot decision boundaries for training and test sets
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Train")
    plot_decision_boundary(model_0, X_train, y_train)
    plt.subplot(1, 2, 2)
    plt.title("Test")
    plot_decision_boundary(model_0, X_test, y_test)
    plt.show()