# GradeWise — Student Performance Predictor

GradeWise is a machine learning web application that predicts a student's final mathematics grade using previous grades and selected study-related factors.

## Features

- Predicts final grades using machine learning
- Interactive web interface built with Streamlit
- Data preprocessing using pandas and scikit-learn
- Model evaluation using MAE, RMSE, and R²
- Displays predictions in a simple interface

## Technologies Used

- Python
- pandas
- scikit-learn
- Streamlit
- NumPy
- Joblib

## Project Structure

```text
Student-Performance-Predictor/
├── app.py
├── train.py
├── requirements.txt
├── README.md
├── .gitignore
├── data/
│   └── student-mat.csv
└── models/
    ├── model.joblib
    └── metadata.json
```

## How to Run

1. Clone or download this repository.
2. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

3. Train the model:

   ```bash
   python train.py
   ```

4. Launch the application:

   ```bash
   streamlit run app.py
   ```

## Dataset

This project uses the Student Performance dataset from the UCI Machine Learning Repository.

Source: https://archive.ics.uci.edu/dataset/320/student+performance

## Disclaimer

Predictions are estimates for educational demonstration purposes only. They should not be used to make important academic decisions about students.

## Future Improvements

- Add more visualizations
- Compare additional machine learning models
- Improve model evaluation
- Deploy the application online

## Author

Developed as a student machine learning project.

## Live Demo

Try GradeWise here:
https://student-performance-predictor-jrxax7d53pmty3ib6ippdl.streamlit.app