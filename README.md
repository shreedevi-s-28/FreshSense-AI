<<<<<<< HEAD
# FreshSense AI

A neural network project that predicts whether a fruit or vegetable is
**Fresh**, should be **Consumed Soon**, or is **Spoiled**, based on its
storage conditions and observable quality indicators.

See [`research.md`](research.md) for the project's background, supported
produce, feature list, target classes, and references.

## How it works

1. **`generate_dataset.py`** synthesizes a labeled dataset. For each row,
   a produce type, storage temperature, humidity, and duration are
   sampled (mixing well-managed and poorly-managed storage scenarios).
   A latent *spoilage index* is computed from how far those conditions
   drift from the produce's normal storage guidelines
   (`constants.PRODUCE_CONFIG`), and the five observable quality scores
   (firmness, color, smell, mold, surface damage) plus the final
   freshness label are derived from that index — so features and labels
   stay internally consistent rather than being assigned independently.

2. **`utils/model_utils.py`** implements a feed-forward neural network
   **from scratch in NumPy** — forward pass, softmax cross-entropy loss,
   and backpropagation are all written out explicitly (no ML framework
   for the core algorithm).

3. **`app.py` + `pages/`** is a Streamlit app to generate/explore the
   dataset, train the model with adjustable hyperparameters, and run
   live predictions.

## Project structure

```
FreshSense-AI/
├── app.py                     # Streamlit landing page
├── constants.py                # produce configs, feature/target schema
├── generate_dataset.py         # synthetic dataset generator
├── requirements.txt
├── research.md                 # project background & references
├── assets/                     # static assets (icons/images) for the app
├── dataset/                    # generated CSV + saved model artifacts
├── pages/
│   ├── 1_Dataset_Overview.py   # explore class balance & distributions
│   ├── 2_Train_Model.py        # train the from-scratch NN, live metrics
│   └── 3_Predict_Freshness.py  # interactive freshness prediction
└── utils/
    ├── data_utils.py           # spoilage simulation, encoding, splitting
    └── model_utils.py          # from-scratch NumPy neural network
```

## Setup

```bash
pip install -r requirements.txt
```

## Usage

**1. Generate the dataset** (defaults to 12,000 rows, well above the
10,000-row target):

```bash
python generate_dataset.py
# or: python generate_dataset.py --n 20000 --seed 7
```

This writes `dataset/freshsense_dataset.csv`.

**2. Launch the app:**

```bash
streamlit run app.py
```

From the sidebar:
- **Dataset Overview** — class balance, per-produce counts, feature
  distributions, and a filterable data table.
- **Train Model** — set epochs/batch size/learning rate/hidden layer
  sizes, train the from-scratch NN, and watch live loss/accuracy curves.
  Saves weights to `dataset/artifacts/`.
- **Predict Freshness** — enter storage conditions and observed quality
  indicators for any supported produce and get a live prediction with
  class probabilities.

## Notes

- The dataset is **synthetic**, generated from storage-guideline
  parameters in `constants.py` rather than scraped or collected from any
  live source — see `research.md` for the food-science references that
  informed those parameters.
- All feature engineering (one-hot produce encoding, numeric
  standardization) lives in `utils/data_utils.py` and is shared by the
  training and prediction pages, so a saved model and new inference rows
  are always encoded the same way.
=======
# FreshSense-AI
Neural network that predicts fruit &amp; vegetable freshness (Fresh / Consume Soon / Spoiled) from storage conditions and quality indicators — synthetic dataset generation, a from-scratch NumPy neural network, and a Streamlit app to explore, train, and predict.
>>>>>>> 6c97b26052c9fdc0e22e13197d63e35bb5976149
