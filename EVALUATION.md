# Evaluation & Limitations

This doc covers how to properly evaluate this model, and the honest limitations of what we built.

## Why accuracy alone isn't enough

```
test set size: 8964
class balance: 1765 toxic / 7208 non-toxic

accuracy:  0.902
baseline:  0.803  (always predicting non-toxic)

precision (toxic class): 0.846
recall    (toxic class): 0.615
f1 score:                0.712

confusion matrix:
                 predicted non-toxic   predicted toxic
  actual non-toxic        7011                 197
  actual toxic             680                1085
```

90.2% accuracy looks great on its own until you notice that always
guessing "not toxic" gets you 80.3% for free, since the dataset is
~80% non-toxic. **Accuracy on imbalanced data can hide a model that's
barely better than doing nothing.**

This model has a real, specific weakness: it misses **680 actually
toxic messages** (false negatives) while only wrongly flagging **197**
non-toxic ones (false positives). It's cautious, not aggressive due to the training data imbalance.


## Known limitations

- **Label imbalance** (~80/20 non-toxic/toxic): realistic for real
  chat, but means accuracy alone is misleading — see above.
- **Recall on toxic messages is the weak point** (61.5%): this model
  under-flags more than it over-flags. Depending on the deployment
  context, that tradeoff might need to go the other way (e.g. a
  stricter moderation system might prefer higher recall even at the
  cost of some false positives).
- **Domain mismatch**: CONDA is Dota 2-specific chat. Slang doesn't
  fully transfer to general context, and need to be considered when building this sort of model
- **Bag-of-words has a hard ceiling**: this representation can't
  capture sarcasm, tone, or anything requiring conversation history.
  Identical text can be genuine banter or a real insult depending on
  context the model never sees. no additional amount of data fixes that,
  it's a structural limit of the choices made when building the model.
