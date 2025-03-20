#  Car Sales Prediction using ANN

##  Project Overview
This project aims to predict the **car purchase amount** based on user demographics using an **Artificial Neural Network (ANN)** built with TensorFlow/Keras. The dataset consists of features such as **age, gender, annual salary, credit card debt, net worth, and country**, which are used to train a predictive model.

We followed a structured **Machine Learning workflow**, including **data preprocessing, feature engineering, ANN training, model evaluation, and predictions on new data**.

---

##  Project Structure
```
car_sales_prediction/
│── car_sales_forecasting.ipynb   # Jupyter Notebook with model training & predictions
│── car_purchasing.csv            # Dataset used for training
│── car_sales_ann.h5              # Trained ANN model
│── README.md                     # Project documentation
│── requirements.txt               # Required Python libraries
```

---

##  Data Processing & Preprocessing

### 🔹 **Step 1: Understanding & Exploring Data**
- Loaded the dataset (`car_purchasing.csv`) into Pandas.
- Explored dataset structure, missing values, and statistical insights.
- Identified numerical and categorical features.

###  **Step 2: Data Preprocessing**
- **Dropped irrelevant columns** (`customer name`, `customer email`).
- **Encoded categorical variables** (`gender`, `country`).
- **Detected and handled outliers** using **IQR method**.
- **Normalized numerical features** using **MinMax Scaling** (to bring all values between 0 and 1).

---

##  Model Development: Artificial Neural Network (ANN)

###  **Step 3: Model Training & Evaluation**
We implemented an **ANN model using TensorFlow/Keras**:
- **Input Layer:** Accepts 6 numerical features.
- **Hidden Layers:** 
  - Layer 1: **64 neurons, ReLU activation**
  - Layer 2: **32 neurons, ReLU activation**
  - Layer 3: **16 neurons, ReLU activation**
- **Output Layer:** 1 neuron (Predicting `car purchase amount`, linear activation)

✅ **Compilation:**
```python
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
```
✅ **Training:**
- **Epochs:** 100
- **Batch size:** 16
- **Validation Data:** 20% test split

✅ **Evaluation Metrics:**
- **Mean Absolute Error (MAE):** 0.02
- **Root Mean Squared Error (RMSE):** 0.024
- **R² Score:** 0.98 (Excellent model fit)

✅ **Saved the trained model:**
```python
model.save("car_sales_ann.h5")
```

---

##  Predictions on New Customer Data
###  **Step 4: Making Predictions**
- **Prepared 100 new customers' data** with randomly generated values.
- **Applied the same preprocessing steps** (encoding & normalization).
- **Loaded the saved ANN model (`car_sales_ann.h5`)** to make predictions.
- **Reversed Min-Max Scaling** to get real purchase amounts.

 **Example Prediction Output:**
```
Customer 1: Predicted Purchase Amount = $353,870.69
Customer 2: Predicted Purchase Amount = $423,609.06
```

---

##  Technologies & Libraries Used

| Category          | Libraries/Technologies |
|------------------|----------------------|
| Programming     | Python 3.x |
| Data Handling   | Pandas, NumPy |
| Visualization   | Matplotlib, Seaborn |
| Machine Learning | Scikit-Learn |
| Deep Learning   | TensorFlow, Keras |
| Model Deployment | GitHub |

---

##  Installation & Usage

### **1️ Clone the Repository**
```sh
git clone https://github.com/your-username/Car-Sales-Prediction.git
cd Car-Sales-Prediction
```

### **2️ Install Dependencies**
Create a virtual environment and install required libraries:
```sh
pip install -r requirements.txt
```

### **3️ Run the Jupyter Notebook**
To train the model or make predictions, open the notebook:
```sh
jupyter notebook car_sales_forecasting.ipynb
```

---

##  Future Improvements
🔹 **Hyperparameter tuning** (optimizer, batch size, epochs) for better accuracy.  
🔹 **Feature engineering**: Adding more economic indicators.  
🔹 **Deploy the model** using Flask, Streamlit, or Hugging Face Spaces.  

---

##  License
This project is open-source and available under the MIT License.

---

##  Contributing
Want to improve this project? Feel free to submit a PR or open an issue! 


