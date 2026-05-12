# Challenge 1: Fix My Broken Model

## The Scenario

You've just joined a team that built a customer complaint classifier for a South African telecom company. The model is supposed to categorise incoming complaints into: **billing**, **network**, **service**, and **fraud**.

Your predecessor left the company last week. They said the model was "working great — 94% accuracy!" But when the team deployed it, it started misclassifying almost everything.

Your job: **find out what's wrong and fix it.**

---

## Setup

### Requirements

```bash
pip install -r requirements.txt
```

### Generate the dataset

```bash
python generate_dataset.py
```

This creates `data/complaints.csv` — a dataset of 2,000 synthetic customer complaints.

### Run the broken model

```bash
python broken_classifier.py
```

You should see output claiming ~94% accuracy. **This accuracy is misleading.**

---

## Your Task

1. **Find the bugs** — There are **5 intentional bugs** in `broken_classifier.py` and the data pipeline. They range from obvious to subtle.

2. **Fix them** — Create a `fixed_classifier.py` that addresses the issues you found.

3. **Document your findings** — For each bug, explain:
   - What the bug is
   - Why it's a problem
   - How you found it
   - How you fixed it

4. **Show the improvement** — Run your fixed model and report the *real* performance.

---

## Hints

- Not all bugs are in the code. Some are in the data.
- "High accuracy" doesn't always mean "good model."
- Look at what features the model is actually using.
- Check the data before you check the model.
- Think about what would happen when this model sees *new* data it hasn't trained on.

---

## What We're Evaluating

| Criteria | What we're looking for |
|----------|----------------------|
| Bugs found | Did you identify the real issues? |
| Quality of fixes | Do your fixes actually resolve the problems without introducing new ones? |
| Explanation quality | Do you understand *why* each bug matters, not just *what* it is? |
| Prompt log | How did you use AI to help you debug and learn? |

> Remember: you'll walk us through your submission in an interview. Make sure you can explain every fix you made.

---

## Files

- `broken_classifier.py` — The buggy classification pipeline
- `generate_dataset.py` — Dataset generator (you can read this to understand the data)
- `requirements.txt` — Python dependencies
- `data/` — Generated after running `generate_dataset.py`
