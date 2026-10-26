# The whole project, explained step by step

This document walks through everything the code does, in order, in plain
language. Every section names the file that does the work, so you can read
the explanation and the code side by side. Numbers quoted here come from
`artifacts/results.json`; if you re-run the pipeline they are regenerated,
and this document is updated to match.

---

## 0. The question

When you move your hand, the muscles in your forearm produce tiny electrical
signals. Electrodes on the skin can pick them up. This is called
**electromyography (EMG)**. The question is:

> Looking only at those electrical signals, can a computer tell which hand
> gesture you are making?

And the three harder versions of the question, which are the ones that
matter for a real device:

1. **Within-subject.** Can it recognise *your* gestures after practising on
   you, on the same day?
2. **Cross-day.** Can it still do it tomorrow, after you have taken the
   sensors off and put them back on?
3. **Cross-subject.** Can it recognise the gestures of a person it has
   never seen?

We also compare three places to put the sensors: the **forearm** (the usual
place, 16 electrodes), the **wrist** (where a watch sits, 12 electrodes),
and **both** together (28).

---

## 1. The data (`emgdec/data.py`)

We use **GrabMyo**, a public dataset from the University of Waterloo
(Canada), published on PhysioNet under a licence that allows reuse with
credit.

- **43 people**, each recorded on **3 different days**.
- Each day, each person made **16 gestures** plus a "rest" position, so
  **17 classes**. The gestures are things like "make a fist", "open hand",
  "point with the index finger", "turn the palm up".
- Each gesture was held for **5 seconds**, and repeated **7 times**. We call
  one 5-second recording a **trial**.
- The electrodes sampled the signal **2048 times per second**, on 32
  channels, of which 28 are real electrodes (4 are placeholders the
  manufacturer leaves empty).

That is 43 × 3 × 17 × 7 = **15,351 trials**, every single one present
(we checked: `docs/LOGBOOK.md`, "Trial audit"). On disk it is 9.6 GB.

`data.py` knows how to read one trial from the files (a format called WFDB)
and how to walk through all of them. It also checks every file against the
official list of checksums, so we know not a single byte was corrupted in
the download.

> **Where the bytes came from.** PhysioNet's own server limits downloads to
> about 30 KB/s per connection, which would have taken 8 hours. PhysioNet
> also publishes the same files on an Amazon S3 bucket with no limit; we
> downloaded from there in 22 minutes. We still verify every file against
> the checksum list from PhysioNet itself, so the mirror is trusted for the
> bytes but not for the truth. (`DECISIONS.md`, "Download mirror")

---

## 2. Cleaning the signal (`emgdec/preprocess.py`)

Raw EMG contains things we do not want:

- very slow drifts (from the electrode sliding on skin, or breathing),
- very fast noise (electronics),
- hum from the electrical mains (the 60 Hz buzz in Canadian power lines).

So each trial goes through three steps:

1. **Band-pass filter 20–450 Hz.** Keep only frequencies between 20 and
   450 cycles per second; that is where muscle signal lives. Think of it as
   an equaliser that turns the bass and the treble all the way down.
2. **Notch at 60 Hz.** A very narrow cut at exactly the mains frequency.
   We did not assume 60 Hz: we measured it. Looking at the frequency content
   of 20 random trials, there is a deep *hole* at exactly 60 Hz, which means
   the recording hardware had already removed it. Our notch is therefore a
   safety net that does nothing if nothing is there.
   (`artifacts/figures/mains_check.svg`)
3. **Halve the sampling rate** to 1024 per second. Since we have thrown away
   everything above 450 Hz, keeping 2048 samples per second is wasteful (a
   rule called Nyquist says you need just over twice the highest frequency,
   i.e. a bit more than 900 per second). This halves the size of everything
   downstream at no cost.

"Zero-phase" in the code means the filter is run forwards and then
backwards, so it does not shift the signal in time.

One thing this step deliberately does **not** do is scale the signal to a
standard size. That happens later, and the reason is important (section 5).

---

## 3. Cutting it into windows (`emgdec/windows.py`)

A model does not look at a whole 5-second trial; it looks at short snippets
and decides for each one. We cut every trial into **windows of 250 ms**
(256 samples at 1024 per second), each one starting **125 ms** after the
previous (so neighbouring windows overlap by half).

Two details:

- We **drop the first half second** of every trial. That is the person
  reacting to the instruction and moving into the gesture; its label would
  be wrong.
- From the remaining 4.5 s we get 35 windows per trial, so
  15,351 × 35 = **537,285 windows** in total.

Each window is a small table: 28 channels × 256 samples. All of them are
stored in one big file on disk (`data/memmap/X.f16`, 7.7 GB) in a format the
computer can read piece by piece without loading it all into memory, plus a
spreadsheet-like table (`meta.parquet`) with one row per window saying which
person, which day, which gesture, which repetition it came from.

That table is what everything else runs on.

---

## 4. Deciding what is "training" and what is "test" (`emgdec/splits.py`)

This is the most important design decision in the project, so it gets the
longest explanation.

To know whether a model has *learned* something rather than *memorised* it,
you train it on some data and test it on data it has not seen. The trap in
EMG is what "not seen" means. Two neighbouring windows from the same
5-second trial overlap by half and look almost identical. If you shuffle all
537,285 windows and keep 20% aside at random as the test, nearly every test
window has a near-twin in the training set. A model that memorised the
training set would score 99% and you would have learned nothing. This
mistake is common enough in published papers that reviewers look for it.

So we never split window by window. We split by a **group** the model has
not seen, and we have three kinds of group, matching the three questions
from section 0:

| Split | Train on | Test on | How many |
|---|---|---|---|
| within_subject | repetitions 1–4 of one person on one day (5 is kept for checking during training) | repetitions 6–7, same person, same day | 129 (43 people × 3 days) |
| cross_day | days 1 and 2 of one person | day 3 of that person | 43 |
| cross_subject | about 31 people | about 8–9 people the model has never seen | 5 folds, every person is tested exactly once |

There is a function `assert_no_leak` that checks, for any split, that no
training window and no test window share the defining group (same
repetition, same day, or same person respectively). It runs in the test
suite on fake data and on the real table, and the training code calls it
before fitting anything. The test suite also contains a deliberately leaky
split to prove the check actually fires.

The "checking during training" set, called **validation**, is always carved
out of the *training* side. It is used to decide when to stop training a
neural network. If it came from the test side, that decision would be
peeking at the exam.

---

## 5. Describing a window with numbers (`emgdec/features.py`)

A window is 28 × 256 = 7,168 raw numbers. The classical approach does not
feed those to a model directly; it first summarises each channel with a
handful of descriptive numbers, called **features**. We use 11 per channel:

| Feature | In words |
|---|---|
| RMS | how strong the signal is on average |
| MAV | the average distance from zero |
| Waveform length | how far a pen would travel if it drew the signal |
| Zero crossings | how often the signal changes sign (a rough "how fast is it wiggling") |
| Slope sign changes | how often it changes direction |
| Hjorth mobility | roughly the dominant frequency |
| Hjorth complexity | how much the frequency content changes within the window |
| Band power 20–50, 50–100, 100–200, 200–450 Hz | how much energy sits in each of four frequency bands |

So each window becomes 28 × 11 = 308 numbers. We compute these once for all
537,285 windows (about 2 minutes) and store them.

**Now the point deferred from section 2.** Before feeding features to a
model we scale each one so it has mean 0 and spread 1 (so that "RMS in
microvolts" and "zero crossings per window" are comparable). The mean and
spread are computed **from the training windows only**, and then applied
unchanged to the test windows. If we computed them over everything, the
test set would have influenced the model's inputs, which is a small leak.
Small leaks add up, so the rule is absolute.

---

## 6. The first model: LDA (`emgdec/baselines.py`)

**Linear Discriminant Analysis** is one of the oldest classifiers there is
and the standard baseline in EMG papers. In one sentence: it finds the
straight boundaries between classes that best separate the training
examples, assuming each class is a blob of roughly the same shape. It has
essentially no knobs, trains in a fraction of a second, and if a fancier
model cannot beat it, the fancier model is not earning its keep.

The one setting we use is "shrinkage", a standard correction for when you
have many features (308) compared to examples; it stops the model from
trusting accidental correlations.

For every one of the 177 splits × 3 electrode placements = 531 runs, the
code fits LDA on the training windows, predicts the test windows, and saves
only a **confusion matrix**: a 17 × 17 table of counts, "the true gesture was
*i* and the model said *j*". Nothing else about the predictions is kept.

---

## 7. Turning counts into scores (`emgdec/metrics.py`, `emgdec/evaluate.py`)

Why save only the confusion matrix? Because every score we report can be
computed from it, and if every training script computed its own accuracy, a
bug in one of them would be invisible. Instead there is **one** definition
of each score, in `metrics.py`, tested against examples worked out by hand
(`tests/test_metrics.py` shows the arithmetic).

- **Accuracy**: correct predictions ÷ all predictions.
- **F1 per class**: a score that is high only if the model both finds most
  examples of a gesture *and* does not label other gestures as it.
- **Macro-F1**: the average F1 over the 17 classes, so a rare class counts
  as much as a common one. (Here classes are balanced, so it tracks
  accuracy closely.)
- **Chance**: what you would score by always guessing one class: 1/17 ≈ 5.9%.

`evaluate.py` reads all the saved run files and builds
`artifacts/results.json`: for each (model, placement, split kind) the mean
and spread across its splits, the summed confusion matrix, and the score of
every individual split. The website renders its tables from that file.
No number on the web page is typed by hand.

---

## 8. What the baseline found

Accuracy, mean ± standard deviation across splits (17 classes, chance 5.9%):

| Electrodes | Within-subject (n=129) | Cross-day (n=43) | Cross-subject (n=5 folds) |
|---|---|---|---|
| Forearm (16) | 82.0 ± 8.9 | 72.2 ± 8.8 | 53.1 ± 2.7 |
| Wrist (12) | 84.7 ± 7.6 | 68.7 ± 11.6 | 40.6 ± 2.7 |
| Both (28) | **93.4 ± 5.7** | **87.0 ± 7.3** | **67.9 ± 2.2** |

What this says, in words:

- **The three questions really are three different difficulties.** Same
  person same day, 93%. Same person, next day, 87%. Stranger, 68%. Each step
  costs real accuracy, which is exactly why a device that works in the lab
  demo can disappoint in the field.
- **Wrist electrodes alone are as good as forearm electrodes alone when the
  model knows you** (84.7 vs 82.0), but **much worse on strangers** (40.6 vs
  53.1). The wrist signal seems more personal: more dependent on the exact
  anatomy under the sensor. That is a useful fact if you are designing a
  watch.
- **Forearm and wrist together are far better than either** (93 vs ~83),
  so they carry different information, not the same information twice.
- **The cross-subject folds agree with each other** (64.5–70.1%), so the
  67.9% is a stable estimate, not luck of the draw.
- The most confused pair is "Hand Close" vs "Lateral Prehension" (a pinch
  grip), which makes anatomical sense: both close the hand.

Figures: `artifacts/figures/accuracy_by_split_lda.svg`,
`confusion_lda_both_*.svg`, `within_subject_per_subject_lda.svg`.

---

## 9. What comes next

- **A small neural network** (a 1-D convolutional network, ~50k parameters)
  that reads the raw 28 × 256 window instead of the 11 summary features,
  trained on the same splits with the same leak checks, so the two models
  are compared on identical exams.
- **Ablations**: what changes with 150 ms or 400 ms windows; how much each
  electrode ring contributes.
- **Export to the browser**: the trained network is converted to a standard
  format (ONNX) and runs live in a web page on held-out test windows, so a
  visitor can watch it classify.

Each of those will get its own section here when it exists.

---

## Reproducing everything

```
make download   # ~10 GB, ~25 minutes from the S3 mirror
make verify     # checks every file against PhysioNet's checksum list
make windows    # ~7 minutes: builds the 537,285 windows
make reproduce  # features → baselines → evaluate → figures, ~5 minutes
make test       # the test suite, including the leak checks
```

Everything is seeded; running it twice gives the same `results.json`.
