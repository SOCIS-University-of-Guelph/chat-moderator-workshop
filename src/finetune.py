"""
Part 2 - Fine-tuning a pretrained model.

Run with:  python src/finetune.py

In Part 1 you built a model from scratch. Here we start from a model that
already learned about language (and toxicity) from other text, then see what
happens when we adapt it to Dota 2 chat.

The plan:
  A. Before:  test the pretrained model on our data, with no training
  B. Freeze the pretrained layers                       <- TODO 1
  C. Train only the classifier head                     <- TODO 2
  D. Unfreeze everything and fine-tune                  <- TODO 3
  E. After:   compare against "before" and your Part 1 model

Most of the code is written for you. Change the settings below and rerun
to see how the results change.
"""

import os
import time

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from vectorizer import load_csv

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Settings - edit these and rerun!
# ---------------------------------------------------------------------------

# Candidates (DistilBERT toxicity classifiers trained on web comments, NOT Dota chat):
#   "martin-ha/toxic-comment-model"
MODEL_NAME = "martin-ha/toxic-comment-model"

# Index of the "toxic" class in the model's output. The script prints the
# model's id2label so you can check it. Our CSV labels use 1 = toxic.
TOXIC_ID = 1

N_TRAIN = 4000        # training messages to use (a small sample keeps runs fast)
N_TEST = None         # test messages to use (None = all of them; smaller = faster)
MAX_LEN = 64          # chat messages are short; smaller = faster
BATCH_SIZE = 32

HEAD_EPOCHS = 2       # epochs with the base model frozen
HEAD_LR = 1e-3
FT_EPOCHS = 2         # epochs with everything unfrozen
FT_LR = 2e-5          # much smaller: we only want to nudge pretrained weights

SEED = 42

# Messages for the "break it" comparison at the end. Add your own!
TEST_MESSAGES = [
    "wow",
    "wow!",
    "gg ez",
    "nice try idiot",
    "n1 bro",
    "uninstall the game",
    "report mid",
    "wow you are SO good at this game, truly amazing",   # sarcasm?
    "y0u are an id1ot",                                  # evasive spelling
]

# Part 1 model results, from EVALUATION.md (trained from scratch on all the data)
PART1_RESULTS = {"accuracy": 0.902, "precision": 0.846, "recall": 0.615, "f1": 0.712}


# ---------------------------------------------------------------------------
# Helper code (already written) - no need to read this line by line
# ---------------------------------------------------------------------------

def make_batches(texts, labels, tokenizer, device, shuffle=False, batch_size=BATCH_SIZE):
    """Tokenize messages and hand them to the model a few at a time.
    (A transformer is too big to run all messages at once like our Part 1 MLP did.)"""
    idx = np.arange(len(texts))
    if shuffle:
        np.random.shuffle(idx)
    for start in range(0, len(idx), batch_size):
        batch_idx = idx[start:start + batch_size]
        inputs = tokenizer(
            [texts[i] for i in batch_idx],
            padding=True, truncation=True, max_length=MAX_LEN,
            return_tensors="pt",
        ).to(device)
        y = torch.tensor([labels[i] for i in batch_idx])
        if TOXIC_ID == 0:          # model's class order is flipped vs our CSV
            y = 1 - y
        yield inputs, y.to(device)


def evaluate(model, tokenizer, device, texts, labels):
    """Accuracy, precision, recall and F1 for the toxic class (same metrics as EVALUATION.md)."""
    model.eval()
    preds = []
    with torch.no_grad():
        for inputs, _ in make_batches(texts, labels, tokenizer, device, batch_size=128):
            logits = model(**inputs).logits
            preds.extend((logits.argmax(dim=1) == TOXIC_ID).long().tolist())

    y_true = np.array(labels)
    y_pred = np.array(preds)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())

    acc = (tp + tn) / len(y_true)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    print(f"  accuracy {acc:.3f} | precision {prec:.3f} | recall {rec:.3f} | f1 {f1:.3f}")
    print(f"  missed toxic (false negatives): {fn} | wrongly flagged (false positives): {fp}")
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


def count_trainable(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  trainable parameters: {trainable:,} of {total:,}")


def toxic_prob(model, tokenizer, device, texts):
    model.eval()
    inputs = tokenizer(
        texts, padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt"
    ).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(**inputs).logits, dim=1)[:, TOXIC_ID]
    return probs.tolist()


def train_epoch(model, optimizer, loss_fn, train_texts, train_labels, tokenizer, device):
    """One pass over the training sample. Same idea as Part 1's train():
    predict, measure how wrong, backpropagate, step."""
    model.train()
    total_loss, n_batches = 0.0, 0
    for inputs, labels in make_batches(train_texts, train_labels, tokenizer, device, shuffle=True):
        optimizer.zero_grad()
 
        # TODO 2: get the model's raw predictions (logits), then the loss.
        # Hint: the model is called on the tokenized inputs like this:
        #       model(**inputs).logits
        # loss_fn compares logits to labels (this batch's true labels), just like in Part 1.
        logits = None
        loss = None
 
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    return total_loss / n_batches


def print_table(results):
    print(f"\n{'model':34s} {'acc':>6s} {'prec':>6s} {'recall':>7s} {'f1':>6s}")
    print("-" * 63)
    for name, r in results.items():
        print(f"{name:34s} {r['accuracy']:6.3f} {r['precision']:6.3f} {r['recall']:7.3f} {r['f1']:6.3f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")

    # --- load data (same CSVs as Part 1) ---
    train_rows = load_csv(os.path.join(SCRIPT_DIR, "..", "data", "train.csv"))
    test_rows = load_csv(os.path.join(SCRIPT_DIR, "..", "data", "test.csv"))

    # small random sample for training; test set stays full so comparisons are fair
    sample = np.random.permutation(len(train_rows))[:N_TRAIN]
    train_texts = [train_rows[i][0] for i in sample]
    train_labels = [train_rows[i][1] for i in sample]

    if N_TEST is not None:
        test_rows = test_rows[:N_TEST]
    test_texts = [t for t, _ in test_rows]
    test_labels = [l for _, l in test_rows]

    print(f"train sample: {len(train_texts)}  test: {len(test_texts)}")

    # --- load the pretrained model ---
    # The tokenizer replaces everything vectorizer.py did in Part 1.
    # The model comes with pretrained weights plus a classification head.
    print(f"\nloading {MODEL_NAME} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).to(device)
    print(f"model labels: {model.config.id2label}  (TOXIC_ID is set to {TOXIC_ID} - does that match?)")

    results = {"Part 1 MLP (from scratch)": PART1_RESULTS}

    # --- A. Before: no training on our data ---
    print("\n=== A. BEFORE: pretrained model, no training on Dota chat ===")
    t0 = time.time()
    results["Pretrained - before"] = evaluate(model, tokenizer, device, test_texts, test_labels)
    print(f"  ({time.time() - t0:.0f}s)")

    # --- B. Freeze the pretrained layers ---
    # Every weight has a switch called requires_grad. When True, optimizer.step()
    # may update it. When False, the weight is frozen.
    # model.base_model is the big pretrained part; the classifier layers sit on top.
    print("\n=== B. FREEZE the pretrained base ===")
    for param in model.base_model.parameters():
        pass  # TODO 1: freeze this parameter (hint: param.requires_grad)
    count_trainable(model)  # should now be a small fraction of the total

    # --- C. Train just the head ---
    print("\n=== C. TRAIN THE HEAD (base frozen) ===")
    loss_fn = nn.CrossEntropyLoss()  # the model outputs 2 logits, one per class
    # only give the optimizer the weights that are allowed to change
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=HEAD_LR
    )
    for epoch in range(HEAD_EPOCHS):
        t0 = time.time()
        avg_loss = train_epoch(model, optimizer, loss_fn, train_texts, train_labels, tokenizer, device)
        print(f"  head-only epoch {epoch + 1} | train loss {avg_loss:.4f} | {time.time() - t0:.0f}s")
    results["Pretrained - head only"] = evaluate(model, tokenizer, device, test_texts, test_labels)

    # --- D. Unfreeze and fine-tune everything ---
    # The pretrained weights are already good, so we only want to nudge them:
    # that's why the learning rate drops (HEAD_LR -> FT_LR).
    print("\n=== D. UNFREEZE and fine-tune everything ===")
    for param in model.base_model.parameters():
        pass  # TODO 3: unfreeze this parameter
    count_trainable(model)  # should now be every parameter

    optimizer = torch.optim.AdamW(model.parameters(), lr=FT_LR)
    for epoch in range(FT_EPOCHS):
        t0 = time.time()
        avg_loss = train_epoch(model, optimizer, loss_fn, train_texts, train_labels, tokenizer, device)
        print(f"  fine-tune epoch {epoch + 1} | train loss {avg_loss:.4f} | {time.time() - t0:.0f}s")

    # --- E. After: compare everything ---
    print("\n=== E. AFTER: fine-tuned model ===")
    results["Pretrained - after fine-tuning"] = evaluate(model, tokenizer, device, test_texts, test_labels)
    print_table(results)

    print("\nThink about it:")
    print("  1. Which model has the best recall on toxic messages? (Part 1's weak spot was 0.615.)")
    print("  2. Did fine-tuning on only a few thousand examples help? By how much?")
    print("  3. Is accuracy alone enough to compare these? Look at precision and recall too.")

    # --- Break it: original vs fine-tuned, head to head ---
    print("\n=== BREAK IT: original pretrained model vs your fine-tuned one ===")
    original = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).to(device)
    before = toxic_prob(original, tokenizer, device, TEST_MESSAGES)
    after = toxic_prob(model, tokenizer, device, TEST_MESSAGES)
    print(f"{'message':52s} {'before':>6s} {'after':>6s}")
    for text, b, a in zip(TEST_MESSAGES, before, after):
        print(f"{text:52s} {b:6.2f} {a:6.2f}")

    print("\nWhich messages changed the most? Which did both versions still get wrong?")
    print("The model still only sees ONE message at a time. What can't it know?")
    print("Try the same messages in your Part 1 demo (python src/app.py) and compare.")


if __name__ == "__main__":
    main()