# Import dependencies
import torch
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split

from torch import nn

# Build model
class BlobModel(nn.Module):
    def __init__(self, input_features, output_features, hidden_units=8):
        """Initializes all required hyperparameters for a multi-class classification model.

        Args:
            input_features (int): Number of input features to the model.
            out_features (int): Number of output features of the model
              (how many classes there are).
            hidden_units (int): Number of hidden units between layers, default 8.
        """
        super().__init__()
        self.linear_layer_stack = nn.Sequential(
            nn.Linear(in_features=input_features, out_features=hidden_units),
            # nn.ReLU(), # <- does our dataset require non-linear layers? (try uncommenting and see if the results change)
            nn.Linear(in_features=hidden_units, out_features=hidden_units),
            # nn.ReLU(), # <- does our dataset require non-linear layers? (try uncommenting and see if the results change)
            nn.Linear(in_features=hidden_units, out_features=output_features), # how many classes are there?
        )

    def forward(self, x):
        return self.linear_layer_stack(x)


if __name__ == "__main__":
    # Set the hyperparameters for data creation
    NUM_CLASSES = 4
    NUM_FEATURES = 2
    RANDOM_SEED = 42

    # 1. Create multi-class data
    X_blob, y_blob = make_blobs(n_samples=1000,
                                n_features=NUM_FEATURES, # X features
                                centers=NUM_CLASSES, # y labels
                                cluster_std=1.5, # give the clusters a little shake up (try changing this to 1.0, the default)
                                random_state=RANDOM_SEED
                                )

    # 2. Turn data into tensors
    X_blob = torch.from_numpy(X_blob).type(torch.float)
    y_blob = torch.from_numpy(y_blob).type(torch.LongTensor)
    print(X_blob[:5], y_blob[:5])

    # 3. Split into train and test sets
    X_blob_train, X_blob_test, y_blob_train, y_blob_test = train_test_split(X_blob,
                                                                            y_blob,
                                                                            test_size=0.2,
                                                                            random_state=RANDOM_SEED
                                                                            )

    # 4. Plot data
    plt.figure(figsize=(10, 7))
    plt.scatter(X_blob[:, 0], X_blob[:, 1], c=y_blob, cmap=plt.cm.RdYlBu)
    plt.show()

    # Create an instance of BlobModel and send it to the target device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_4 = BlobModel(input_features=NUM_FEATURES,
                        output_features=NUM_CLASSES,
                        hidden_units=8).to(device)
    model_4
