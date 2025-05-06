# Hotel Booking Cancellation Prediction

## Description
This project analyzes a hotel booking dataset and builds an Artificial Neural Network (ANN) to predict whether a booking will be canceled. It covers exploratory data analysis (EDA), feature engineering (including mapping countries to continents), data preprocessing, model training, and evaluation.

## Features
- **Data Loading**: Reads the “hotel_bookings.csv” dataset from local or mounted drive.
- **Exploratory Data Analysis**:  
  - Univariate analysis of categorical and numerical features  
  - Multivariate analysis against the cancellation target  
  - Visualization of country-to-continent distribution  
- **Preprocessing Pipeline**:  
  - Handling missing values with `SimpleImputer`  
  - Scaling numerical features with `StandardScaler`  
  - Encoding categorical variables (one‑hot and label mapping)  
- **Modeling**:  
  - Builds a Keras Sequential ANN with three hidden layers (128 → 64 → 32) and sigmoid output  
  - Trains for 100 epochs with binary crossentropy loss and Adam optimizer
- **Evaluation**:  
  - Reports accuracy, confusion matrix, and classification metrics on a held‑out test set

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/hotel-booking-cancellation-predictor.git
   cd hotel-booking-cancellation-predictor
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Place the `hotel_bookings.csv` file in the project root (or update the path in the notebook/script).
2. Run the analysis notebook or main script:
   ```bash
   jupyter notebook
   # or
   python train_model.py
   ```
3. Follow the notebook steps or script logs to reproduce EDA, preprocessing, training, and evaluation.

## Project Structure

```
├── data/
│   └── hotel_bookings.csv        # Raw dataset
├── notebooks/
│   └── EDA_and_Model.ipynb       # Jupyter notebook with full workflow
├── train_model.py                # Script to preprocess data and train ANN
├── requirements.txt              # Python dependencies
└── README.md                     # Project overview and instructions
```

## License
This project is released under the MIT License. Feel free to use and modify.

## Author
Ahmed Ibrahim Khazani
