import json
import os

import torch
import torch.nn as nn

from model import ChatModerator
from vectorizer import BigramVectorizer, load_csv

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Fixed seed for reproducibility. Without this, nn.Linear's random weight
# initialization differs every run
torch.manual_seed(42)


def train(model, X_train, y_train, X_test, y_test, epochs=20, lr=0.01):
    """
    Standard PyTorch training loop.

    X_train / X_test: bag-of-uni+bigrams vectors, shape (num_examples, vocab_size)
    y_train / y_test: labels, shape (num_examples, 1) — 1.0 = toxic, 0.0 = not
    """

    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        # --- training step ---
        model.train()  

        optimizer.zero_grad()          # clear gradients from the previous step

        # TODO: Make predictions on the training set, then measure how wrong those predictions were using the function loss_fn
        logits = None
        loss = None

        loss.backward()                # backpropagate to compute gradients
        optimizer.step()               # nudge every weight downhill according to its corresponding gradient

        # --- evaluation step against test set ---
        model.eval()  # eval mode
        with torch.no_grad():  # don't compute gradients
            train_preds = (torch.sigmoid(logits) > 0.5).float()
            train_acc = (train_preds == y_train).float().mean().item()

            test_logits = model(X_test)
            test_loss = loss_fn(test_logits, y_test)
            test_preds = (torch.sigmoid(test_logits) > 0.5).float()
            test_acc = (test_preds == y_test).float().mean().item()

        print(
            f"epoch {epoch+1:2d} | "
            f"train loss {loss.item():.4f}  train acc {train_acc:.2f} | "
            f"test loss {test_loss.item():.4f}  test acc {test_acc:.2f}"
        )

    return model



#when you run this file directly, load and vectorize the data, then train the model


if __name__ == "__main__":
    # --- load real data ---
    train_rows = load_csv(os.path.join(SCRIPT_DIR, "..", "data", "train.csv"))
    test_rows = load_csv(os.path.join(SCRIPT_DIR, "..", "data", "test.csv"))

    train_texts = [text for text, label in train_rows]
    train_labels = [label for text, label in train_rows]

    test_texts = [text for text, label in test_rows]
    test_labels = [label for text, label in test_rows]

    # --- vectorize ---
    # fit on TRAIN ONLY — the vocab is built from training text alone
    vectorizer = BigramVectorizer()

    vectorizer.fit(train_texts, min_count=3)

    X_train = torch.tensor(vectorizer.transform_batch(train_texts), dtype=torch.float32)
    X_test = torch.tensor(vectorizer.transform_batch(test_texts), dtype=torch.float32)

    # labels need shape (num_examples, 1) to match the model's output shape
    y_train = torch.tensor(train_labels, dtype=torch.float32).unsqueeze(1)
    y_test = torch.tensor(test_labels, dtype=torch.float32).unsqueeze(1)

    print(f"vocab size: {vectorizer.vocab_size}")
    print(f"train examples: {X_train.shape[0]}  test examples: {X_test.shape[0]}")
    print()

    # --- build + train ---
    # vocab_size now comes from the vectorizer, not a hardcoded guess —
    # this is the "read the shapes, don't hardcode" lesson in practice
    model = ChatModerator(vocab_size=vectorizer.vocab_size, hidden_dim=64)
    train(model, X_train, y_train, X_test, y_test, epochs=20, lr=0.01)

    torch.save(model.state_dict(), os.path.join(SCRIPT_DIR, "model.pt"))
    vectorizer.save(os.path.join(SCRIPT_DIR, "vocab.json"))
    print("\nsaved trained weights to model.pt")
    print("saved fitted vocab to vocab.json")
