# Chat Moderator

A binary toxicity classifier built for SOCIS's AI/ML Workshop series
Session 1.  
Bag-of-(uni+bi)grams vectorization → small PyTorch MLP → Gradio front end, trained on human-annotated game chat.

## Setup

### 1. Clone the repo

```
git clone https://github.com/SOCIS-University-of-Guelph/chat-moderator-workshop.git
cd chat-moderator-workshop
```

### 2. Check your Python version

This workshop requires Python 3.10+ to run. Check your version with ```python --version``` and upgrade if needed.

### 3. Create a virtual environment

```
python -m venv venv
```

### 4. Activate it

**Windows (PowerShell):**
```
venv\Scripts\activate
```

**Mac / Linux:**
```
source venv/bin/activate
```

### 5. Install PyTorch

PyTorch needs its own install command

```
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### 6. Install everything else (Gradio for the front end, NumPy as it is a light dependency of Gradio)

```
pip install -r requirements.txt
```


## Files

| File | What it does |
|---|---|
| `model.py` | The `ChatModerator` class — a 2-layer MLP (linear → ReLU → linear) |
| `vectorizer.py` | Converts text into bag-of-(unigram+bigram) count vectors |
| `train.py` | The training loop; loads data, trains, saves `model.pt`/`vocab.json` |
| `app.py` | Gradio front end |
| `finetune.py` | Pre-trained model from HuggingFace |

## Dataset

This project uses the **CONDA dataset** (Weld et al., 2021) — 45K
real, human-annotated utterances from Dota 2 chat.

- Source: https://github.com/usydnlp/CONDA
- Paper: https://arxiv.org/abs/2106.06213

`intentClass` labels (`E`=explicit toxicity, `I`=implicit toxicity,
`A`=action, `O`=other) are collapsed to binary: toxic = E or I,
non-toxic = A or O.

The data was cleaned and transformed for our uses, using a script that loaded and cleaned them at once.


# Part 1 - Building and Training a Model From Scratch

Your goal is to build the missing pieces allowing the program to turn raw text into information a model can learn from, allowing it to be trained to detect malicious messages in chatrooms.

The point of today isn't to understand the code line-by-line, it's connecting concepts to implementation. You will be focusing on fixing issues in two files, ```vectorizer.py``` and ```train.py``` in that order.

```vectorizer.py``` - Learning how to turn words into numbers the model can understand.

```train.py``` - Determining if the results you gain is the answer you were looking for.

## Step 1 - ```get_ngrams()``` in ```vectorizer.py```

As discussed previously, the model needs the chat messages to become numbers in order to be able to understand it. The approach here is called bag-of-n-grams: count which words/word-pairs show up, and disregard anything about order or grammar.

You will have to fill in two lists:

- Unigrams: one tuple per <u>individual</u> word - ```["hello", "world"]``` turns into ```[("hello",), ("world",)]```.

- Bigrams: one tuple per <u>adjacent</u> word - ```["hello", "world"]``` turns into ```[("hello", "world")]```.

<details>
<summary> <b>Hint</b> </summary>

>You need to turn ```tokens``` (a list of strings) into a list of 1-tuples (1-tuple example: ```("SOCIS",)```, notice the comma). For each token, think of how you can format it like a 1-tuple, then append it to ```unigrams```.

>For bigrams, the loop variable ```i``` is already set up to range over every valid starting positions for a pair. For each value of ```i```, build a 2-tuple out of the word at position ```i``` and the word right after it, then append it to the ```bigrams``` list.

</details>

## Step 2 - ```fit()``` in ```vectorizer.py```

Now that we are able to break a message into n-grams, ```fit()``` needs to do two things:

1. Count frequencies: Track how often each n-gram appears across the entire training set.

2. Build the index: Assign a unique integer ID to every n-gram that meets the ```min_count``` threshold.

By counting occurrences across the dataset, we can filter out rare n-grams like typos, names, or other irrelevant phrases. By doing this, we prevent the vocabulary from being massive, and keep the model focused on patterns that are useful to our model.

<details>
<summary> <b>Hint</b> </summary>

>For counting, use Python's dict.get() method to look up an n-gram's current count (automatically defaulting to ```0``` if it hasn't been seen yet), and incrementing it by ```1```. 

>More info on dict.get(): https://www.w3schools.com/python/ref_dictionary_get.asp

>For indexing, ```len(self.bigram_to_index)``` gives you the current size of the vocabulary. This automatically serves as the next available index number.

</details>

## Step 3 - ```transform()``` in ```vectorizer.py```

Our previous steps had us counting n-grams while building the vocabulary for the model. Here, the vocabulary is fixed, and now we are converting a brand new message into a vector, using the vocabulary we built previously.

<details>
<summary> <b>Hint</b> </summary>

>Instead of building a running total, now you're looking something up in what was previously built. If this n-gram showed up often enough during training, it will have a slot in ```self.bigram_to_index```. If it didn't show up enough to be included in the vocabulary, it needs to fall into slot 0 (```UNK```). Think back to how we counted how many times an n-gram appeared across the whole training set, while defaulting to a number if it's not there back in Step 2.

</details>

## Checkpoint - Run ```vectorizer.py``` on its own

Once you have completed the five ```TODOs``` found in ```vectorizer.py```, try running it by itself and see what it outputs:

```python src/vectorizer.py```

Your output should look something close to:

```
vocab size (including UNK): 53696

sample train message: 'wow!'
vector length: 53696
nonzero slots: 1

sample test message: 'GG'
UNK count in this message's vector: 0
```

A couple of things worth noticing here:

Only 1 nonzero slot for "wow!". The tokenizer just lowercases and splits on spaces, it doesn't strip punctuation. So "wow!" and "wow" are two different n-grams as far as the model is concerned. Keep that in mind later if your own test messages behave unexpectedly.

UNK count: 0 for the test message "GG" - that means "gg" already showed up in training, so it's not a stranger to the vocabulary. Try changing the sample index in the __main__ block to look at a different test message. Can you find one that does have UNKs?

Heads up: when you get to train.py, the vocab size printed there will be noticeably smaller than the ~53,696 you saw here. That's because train.py calls fit(train_texts, min_count=3) instead of the default min_count=1, dropping any n-gram that showed up fewer than 3 times across the whole training set. That's the "keep out the long tail of typos and one-off phrases" idea from the fit() docstring in action.


## Step 4 - ```train()``` in ```train.py```

This is the part that actually teaches the model. Think of these guiding questions:

1. Is this the answer I wanted?

2. If not, how far off was it?

3. How should the function adjust itself to do better next time?

Your job is to code the lines that solve questions 1 and 2. Question 3 is already written, being ```loss.backward()``` and ```optimizer.step()```. These filled in lines are right below where your task lies. 


<details>
<summary> <b>Hint</b> </summary>

>You already have a working model, it's an object you can call like a function. Calling it on the training inputs gets you the raw guesses, called "logits". Now you just need something that compares the model's guesses to the actual labels, and produces a number representing how wrong the guess was (this is what the ```loss_fn``` object is for)

</details>

## Checkpoint - Run ```train.py```

```python src/train.py```

You should see 20 lines of output, one per epoch. The most important thing among the output you should notice is a **trend** as epochs go up: 

- ```train loss``` and ```test loss``` should generally go down
- ```train acc``` and ```test acc``` should generally go up

Notably, the accuracy should end up noticeably above 0.80. Since ~80% of messages on the dataset are labeled "not toxic", a model that blankly guesses "not toxic" on all messages should get an accuracy of around 0.80. If the final accuracy doesn't end up higher than that, the model didn't actually learn much.

If loss is flat or accuracy never moves, that's a sign Step 4 isn't wired up correctly yet. It usually means logits or loss still isn't a real number. Check that you're calling the model itself for one, and ```loss_fn``` for the other.

## Knowledge Check - What does ReLU do?

Now that you have a working model, let's poke at one piece of it. Open ```model.py``` and find this piece of code on Line 17:

```
self.relu = nn.ReLU()
```

ReLU is the activation function sitting between the model's two linear layers. What happens when we take it away? Let's find out!

1. Take note of your final values for `train loss`, `train acc`, `test loss`, and `test acc` from the run of ```train.py``` you just did, which included the ReLU function (write the values down, or just keep them in your terminal)

2. In ```model.py```, swap the ReLU code on Line 17 for:

```
self.relu = nn.Identity()
```

(```nn.Identity()``` returns whatever it's given, so the model still runs, but ReLU no longer does anything)

3. Before running, make a prediction: Will accuracy go up, down, or stay the same? What about the loss?

4. Run `python src/train.py` again and compare the two runs.

> **Heads up:** `train.py` overwrites `model.pt` when it finishes. When you're done experimenting, change `model.py` back to `nn.ReLU()` and run `train.py` once more before the demo, so the saved weights match the model.

How does the final **test accuracy** compare between the two runs? What about the **test loss**?

Here's a question to think about: without ReLU, `fc1` and `fc2` are two linear layers back to back. Why is that no more powerful than a single linear layer?

<details>
<summary><b>Answer (try it first!)</b></summary>

>A linear transformation followed by another linear transformation is still just one linear transformation, so stacking them adds no expressive power. ReLU is the non-linear step between them that lets the network learn more than a straight-line relationship. 

>Think of it like a sandwich, with the two linear layers being the bread, and ReLU being the filling. Two slices of bread stacked together is just thicker bread! ReLU is the non-linear step in the middle of the two that makes the layers more than the sum of their parts.

</details>

Don't be surprised if your two runs are very close. This dataset is mostly about which words and word-pairs appear, which a linear model handles well. A non-linear step matters more on harder problems where features interact, like sarcasm or tone. More complexity doesn't automatically mean better results. 

**Think about it:** what kinds of messages might a purely linear, model that is blind to order struggle with? Keep this in mind for the next step.

## Final Step - Run the demo

1. Run the demo: ```python src/app.py```.

2. Break it. Try to get it wrong on purpose. Typos, evasive spelling, sarcasm, creative punctuation. When you find something that fools it, ask yourself why. The answer usually connects back to something in this guide (hint: bag-of-words can't see word order or tone at all).

3. Read EVALUATION.md. Compare the failure case you found by hand against the model's actual weak spot in the numbers (recall on the toxic class). Are they the same kind of failure, or different?



# Part 2 - Fine-tuning a Pretrained Model

In Part 1 you built a model from scratch: random weights, your own vectorizer. Here we start from a model that already learned about language (and toxicity) from *other* text, then adapt it to Dota 2 chat. The training loop is the same idea as before. What changes is where the weights start.

Like Part 1, the point isn't to understand every line, it's connecting concepts to implementation. You will be working in one file, ```finetune.py```, in three places.

```finetune.py``` - Taking a pretrained model, deciding which of its weights are allowed to change, and training it on our data.

## Before you start

Install the one extra package (PyTorch is already installed from earlier):

```
pip install -r requirements-part2.txt
```

The first run downloads the pretrained model (about 270 MB), so it may take a little to download. It runs on CPU, and a full run can take around 10 minutes.

The settings at the top of `src/finetune.py` (training sample size, epochs, learning rates, test messages) are meant to be edited, so feel free to change them and rerun.


## Step 1 - Freezing the pretrained layers in ```finetune.py```

A pretrained model comes in two parts: a big pretrained body that understands language, and a small classifier head on top. Every weight in a PyTorch model has a switch called ```requires_grad```. When it's ```True```, ```optimizer.step()``` is allowed to update that weight. When it's ```False```, the weight is **frozen**.

Find ```TODO 1``` in ```main()```. ```model.base_model``` is the pretrained body. Your job is to freeze every one of its parameters.

<details>
<summary> <b>Hint</b> </summary>

>Each ```param``` in the loop has a ```requires_grad``` attribute. Set it to ```True``` or ```False```.

</details>

## Step 2 - ```train_epoch()``` in ```finetune.py```

This is the same job as ```train()``` in Part 1: get the model's predictions, then measure how wrong they were. Question 3 from before (how to adjust) is already written as ```loss.backward()``` and ```optimizer.step()```, right below your task.

Find ```TODO 2```. You need to fill in ```logits``` and ```loss```.

> **Note:** ```train_epoch()``` is defined above ```main()``` in the file, so you'll run into ```TODO 2``` before ```TODO 1```. The TODOs are numbered in the order the program runs, so follow the numbers in the comments.

<details>
<summary> <b>Hint</b> </summary>

>The model is an object you can call like a function. This time, the inputs are already tokenized for you, so call it like this: ```model(**inputs).logits```.

>Then, just like Part 1, ```loss_fn``` compares the logits to the true ```labels``` and gives you a number representing how wrong the guess was.

</details>

## Step 3 - Unfreezing everything in ```finetune.py```

Training only the head is fast, but it can only adjust the very last layers. Now we let *every* weight adapt to Dota chat. Find ```TODO 3```, which is the opposite of Step 1.

Notice the learning rate drops here (```FT_LR``` is much smaller than ```HEAD_LR```). The pretrained weights are already good, so we only want to nudge them.

<details>
<summary> <b>Hint</b> </summary>

>It's the same loop as Step 1, with the switch flipped the other way.

</details>

## Checkpoint - Run ```finetune.py```

```python src/finetune.py```

The script goes through five stages: testing the model **before** any training, **freezing** it, training just the **head**, **unfreezing** and fine-tuning everything, then testing **after**.

After Step 1, the number of trainable parameters should drop to a small fraction of the total. Your output should look something close to:

```
=== B. FREEZE the pretrained base ===
  trainable parameters: 592,130 of 66,955,010
```

After Step 3, it should go back up to the full total. At the very end you'll see a table comparing your Part 1 model with each stage here, something close to:

```
model                                 acc   prec  recall     f1
---------------------------------------------------------------
Part 1 MLP (from scratch)           0.902  0.846   0.615  0.712
Pretrained - before                 0.810  0.573   0.127  0.208
Pretrained - head only              0.810  0.560   0.138  0.221
Pretrained - after fine-tuning      0.923  0.842   0.746  0.791
```

A couple of things worth noticing here:

The "before" accuracy is barely above 0.80, which is what a model that always guesses "not toxic" would get. This model learned from general web comments, so Dota chat is unfamiliar to it. Look at the recall column to see how much toxic chat it misses.

If the trainable count doesn't change after Step 1, or ```loss``` is ```None``` when you run it, one of the TODOs still has its placeholder in it.

## Knowledge Check - Why didn't freezing help much?

Compare the **head only** row to the **after fine-tuning** row, then think about these:

1. Which model has the best **recall** on toxic messages? Remember that recall was the weak spot in Part 1.

2. Training only the head barely changed the results, but unfreezing everything changed them a lot. Why might freezing alone not be enough?

3. Fine-tuning only used a small sample of the training data, while Part 1 used all of it. What does that tell you about starting from pretrained weights?

<details>
<summary><b>Answers (try them first!)</b></summary>

>1. The fine-tuned model, which should beat both the "before" row and your Part 1 model on recall.

>2. The frozen body was trained on general web text, so the way it represents Dota chat wasn't a good fit. A small head on top can only rearrange what the body gives it. Letting the body's weights change is what lets the model adapt to a new domain.

>3. Starting from pretrained weights means the model already knows a lot about language, so it needs much less of your data to adapt than a model starting from random weights.

</details>

## Final Step - Break it, again

1. Look at the **break it** table at the bottom of the output. It compares the original pretrained model to your fine-tuned one on the same messages.

2. Add your own messages to ```TEST_MESSAGES``` at the top of ```finetune.py``` and rerun. Try typos, evasive spelling, sarcasm, and creative punctuation, like in Part 1. Which messages changed the most after fine-tuning? Which did both versions still get wrong? A model that only sees one message at a time can't know about what?

3. Run the same messages through the Part 1 demo (```python src/app.py```). Where do the two approaches disagree, and why?

4. Optional: change ```N_TRAIN``` or ```FT_EPOCHS``` and see how the results move.

