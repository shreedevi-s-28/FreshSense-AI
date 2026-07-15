# FreshSense-AI

FreshSense-AI is a machine learning project that predicts whether a fruit or vegetable is **Fresh**, **Consume Soon**, or **Spoiled** based on storage conditions and observable quality indicators.

## Features

- Synthetic dataset generation (10,000+ samples)
- Neural network implemented **from scratch using NumPy**
- Interactive Streamlit web application
- Live freshness prediction with confidence scores
- Dataset exploration and visualization
- Customizable model training interface

## Tech Stack

- Python
- NumPy
- Pandas
- Streamlit
- Matplotlib

## Project Structure

```
FreshSense-AI/
├── app.py
├── generate_dataset.py
├── constants.py
├── research.md
├── dataset/
├── pages/
└── utils/
```

## Getting Started

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Project Workflow

1. Generate a synthetic dataset.
2. Train a neural network built from scratch using NumPy.
3. Explore the dataset through the Streamlit dashboard.
4. Predict the freshness of fruits and vegetables using storage and quality inputs.

## License

This project is for educational and research purposes.
