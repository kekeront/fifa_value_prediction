# ⚽ FIFA Player Value Prediction

This project predicts the **market value of FIFA players** using machine learning techniques applied on in-game statistics and personal attributes.

---

## 🚀 Project Overview
The objective of this project is to develop a **Random Forest Regressor model** that estimates a player's market value (`value_euro`).  
The workflow includes:
1. Data preprocessing (dropping irrelevant columns, imputing missing values)  
2. Feature engineering (position grouping, handling categorical variables)  
3. Outlier detection and removal  
4. Feature importance analysis  
5. Model training and evaluation  
6. Visualization of results  

---

## 📊 Dataset
- **Source**: `fifa_players.csv`  
- Focus on **club performance**, excluding national leagues and national team performance.  
- Dataset had inconsistencies such as:  
  - Missing ranges in features (e.g., **height_cm** lacked players between 175–185 cm, disrupting distributions).  
  - Many irrelevant or weakly correlated features.  

---

## ⚙️ Data Preprocessing

### Dropped Features
- **Personal info** (not relevant for market value):  
  `name, full_name, birth_date`  
- **National team info** (performance differs across roles):  
  `national_team, national_rating, national_team_position, national_jersey_number`  
- **Weak correlation (<0.01)** with target or redundant:  
  `preferred_foot, body_type, tier_encoded, weak_foot`  
- **Too strong features** (>0.4 feature importance → overfitting risk):  
  `reactions, potential`  
- **Height** (kept only for goalkeepers, since it’s strongly correlated with weight)  

---

### Feature Engineering
- **Positions Encoding** → grouped into 4 categories:  
  - **Attackers**: ST, CF, LW, RW  
  - **Midfielders**: CAM, CM, CDM, LM, RM  
  - **Defenders**: CB, LB, RB, LWB, RWB  
  - **Goalkeepers**: GK  

- **Logarithmic Transformation** applied to skewed features with exponential trends to improve linearity.  

---

### Outlier Handling
- Outliers identified using the **IQR (15–85 percentile)** and removed.  
- Reduced dataset from ~18k players to ~17k players.  

---

## 📈 Feature Importance

Top features with importance > 0.01:  
dribbling → 0.301
positioning → 0.261
finishing → 0.116
short_passing → 0.072
shot_power → 0.032
age → 0.030
sliding_tackle → 0.021
sprint_speed → 0.013


📉 High co-dependence found between certain features (e.g., **height vs weight**). Features were grouped or reduced to prevent overfitting.

---

## 📊 Visualizations

Some example visualizations used in the analysis:

- Distribution of player values  
- Correlation heatmap of features  
- Feature importance ranking  
- Residuals vs predicted values  
- Before/after logarithmic transformation of skewed features  

Example residuals plot:  
<Figure size 1000x600 with 1 Axes><img width="844" height="547" alt="image" src="https://github.com/user-attachments/assets/5bf68771-937b-4a3b-bd02-817a88740bf1" />



Feature importance example:  
![Residuals](https://github.com/user-attachments/assets/cf076e43-93c3-4d3e-92ed-e73b1a6df634)


Correlation clustering example:  
![Feature Importance](https://github.com/user-attachments/assets/67917f5b-7ed8-42aa-ad63-a92dbedf763b)
![Correlation](https://github.com/user-attachments/assets/0d7d6207-8c48-4248-9b8c-422f1c2b634d)

---

## 🤖 Model Training
We trained a **Random Forest Regressor** with the processed dataset:

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Mean Squared Error: {mse}")
    print(f"R^2 Score: {r2}")
    return model, X_train, X_test, y_train, y_test, y_pred
