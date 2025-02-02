# air-quality-forecasting

# Beijing Air Quality Forecasting with RNNs and LSTM

This repository contains an implementation of an RNN-based model (using LSTMs) for forecasting PM2.5 air pollution levels in Beijing. The model utilizes historical air quality data and meteorological features to predict future PM2.5 concentrations, aiding in environmental monitoring and policy-making.

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Data Preprocessing](#data-preprocessing)
- [Model Architecture](#model-architecture)
- [Training and Evaluation](#training-and-evaluation)
- [Results and Visualizations](#results-and-visualizations)
- [Usage](#usage)
- [References](#references)

---

## Introduction

Air pollution is a significant public health concern, and forecasting PM2.5 levels can help mitigate its adverse effects. This project applies a deep learning approach using Long Short-Term Memory (LSTM) networks to predict PM2.5 concentrations in Beijing.

## Dataset

The dataset used for this project is the **Beijing PM2.5 Data** from the UCI Machine Learning Repository. It contains hourly air quality data from the US Embassy in Beijing, including features such as temperature, humidity, wind speed, and atmospheric pressure.

### Features:
- `datetime`: Timestamp of the recorded data.
- `PM2.5`: Target variable (concentration in micrograms per cubic meter).
- `DEWP`: Dew point temperature.
- `TEMP`: Ambient temperature.
- `PRES`: Atmospheric pressure.
- `cbwd`: Categorical variable indicating wind direction.
- `Iws`: Cumulative wind speed.
- `Is`: Cumulative hours of snow.
- `Ir`: Cumulative hours of rain.

## Project Structure

```
├── data/
│   ├── train.csv        # Training dataset
│   ├── test.csv         # Test dataset
│   └── processed/       # Preprocessed data
├── experiments/
│   ├── Experiment 1.png  # Many experiments photos included
├── models/
│   ├── best_model.h5    # Saved trained model
│   ├── model_architecture.png  # Model visualization
├── notebooks/
│   ├── model_training.ipynb  # Main Jupyter Notebook
├── submissions/
│   ├── submission16.csv  # Many submission file included
├── README.md                   # Project Overview
```



## Data Preprocessing

- **Convert `datetime` to a datetime object** and set it as the index.
- **Handle missing values** using mean imputation.
- **Normalize numerical features** to improve model performance.
- **One-hot encode categorical variables** like wind direction (`cbwd`).
- **Reshape data** to fit LSTM input format `(timesteps, features)`.

## Model Architecture

The final model consists of:
- **LSTM Layers:**
  - First LSTM layer with 128 units, `tanh` activation, dropout of 0.2, and recurrent dropout of 0.2.
  - Second LSTM layer with 64 units, batch normalization, and dropout.
  - Third LSTM layer with 32 units, batch normalization, and dropout.
- **Dense Layers:**
  - A Dense layer with 16 units and ReLU activation.
  - Dropout (0.2).
  - Final Dense layer with 1 unit for regression output.

### Model Visualization
The model architecture is saved as `model_architecture.png`.

## Training and Evaluation

- **Optimizer:** Adam
- **Loss Function:** Mean Squared Error (MSE)
- **Callbacks:**
  - `EarlyStopping` (monitoring validation loss)
  - `ModelCheckpoint` (saving the best model)
  - `ReduceLROnPlateau` (adaptive learning rate reduction)

## Results and Visualizations

- **Loss Curves:** Training vs. Validation loss.
- **Actual vs. Predicted PM2.5 Values:** A time-series plot comparing ground truth vs. model predictions.
- **Residual Analysis:** Histogram of residuals to check the error distribution.


## References

1. World Health Organization, “Ambient (outdoor) air quality and health,” WHO Fact Sheet, 2018.
2. Hochreiter & Schmidhuber, “Long Short-Term Memory,” *Neural Computation*, 1997.
3. I. Goodfellow, Y. Bengio, A. Courville, *Deep Learning*, MIT Press, 2016.



## Contact

For any questions or further information, please contact [e.bakunzi@alustudent.com](mailto:e.bakunzi@alustudent.com).

