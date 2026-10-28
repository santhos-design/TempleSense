# TempleSense AI - Temple Crowd Prediction (Fundamentals of AI project)

Predicts the crowd level (Low / Moderate / High / Very High) at 20 Tamil Nadu temples for a chosen date using a
Random Forest classifier, and converts it into a best-darshan-time recommendation.

## Run
```
pip install -r requirements.txt
python generate_data.py     # creates data/*.csv  (synthetic, seeded)
python train_model.py       # trains models, saves model/crowd_rf.joblib, model/metrics.json, docs/*.png
python app.py               # open http://127.0.0.1:5000
python -m unittest tests -v # run tests
```
## Files
- features.py - Tamil calendar / festival / holiday / weather feature engineering (shared)
- generate_data.py - builds temples.csv, festival_calendar.csv, holidays.csv, crowd_data.csv
- train_model.py - Random Forest + Decision Tree, KNN, Logistic Regression comparison, evaluation, charts
- app.py, templates/, static/ - Flask API + dashboard
- tests.py - unit and API tests

Note: the dataset is synthetic (formula-based); lunar days (Amavasai, Pournami, Ekadasi, Pradosham) are
approximated from the moon's phase, so results demonstrate the AI pipeline, not real-world accuracy.
