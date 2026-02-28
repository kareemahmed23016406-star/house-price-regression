# House Price Prediction (Regression)

**Dataset:** Kaggle (place CSV as `data/data.csv`)

This project builds regression models to predict house prices using:
- Linear Regression
- Polynomial Regression (degree=2)
- KNN Regressor (baseline)

## Run
```bash
pip install -r requirements.txt
python src/main.py --data_path data/data.csv
```

## Notes
- The script keeps your original approach: selecting features with `.iloc[:, 3:]` and target with `.iloc[:, 1:2]`.
- It converts `floors` into a binary feature (1 -> 0, else -> 1) as in your notebook.
"# house-price-regression" 
