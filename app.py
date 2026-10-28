"""TempleSense AI - Flask backend. Run:  python app.py  ->  http://127.0.0.1:5000"""
import datetime as dt, json, urllib.parse
import joblib
from flask import Flask, jsonify, render_template, request
from features import *

app = Flask(__name__)
MODEL = joblib.load("model/crowd_rf.joblib")      # loaded once at startup
METRICS = json.load(open("model/metrics.json"))

ADVICE = {
    "Low": ("Anytime between 10:00 AM - 12:00 PM or 5:00 - 7:00 PM", "10 - 20 min", "Calm day - ideal for a peaceful darshan."),
    "Moderate": ("Early morning 6:00 - 8:00 AM or after 6:30 PM", "30 - 45 min", "Manageable crowd; avoid the late-morning rush."),
    "High": ("Before 6:00 AM or after 8:00 PM", "60 - 90 min", "Expect queues; carry water and plan extra time."),
    "Very High": ("Reach before 5:00 AM, or choose another date", "2 - 3 hours", "Peak crowd - consider the lighter day suggested below."),
}


def predict(temple, d):
    X = encode(build_row(temple, d))
    proba = MODEL.predict_proba(X)[0]
    classes = list(MODEL.classes_)
    probs = {LEVELS[c]: float(p) for c, p in zip(classes, proba)}
    return max(probs, key=probs.get), probs


@app.route("/")
def home():
    return render_template("index.html", temples=TEMPLE_NAMES, districts=DISTRICTS,
                           today=dt.date.today().isoformat(), acc=round(METRICS["models"]["Random Forest"]["accuracy"] * 100, 1))


@app.route("/api/temples")
def temples():
    district = request.args.get("district", ""); q = request.args.get("q", "").lower()
    out = [dict(name=n, **TEMPLE_INFO[n]) for n in TEMPLE_NAMES
           if (not district or TEMPLE_INFO[n]["district"] == district) and q in n.lower()]
    return jsonify(out)


@app.route("/api/predict")
def api_predict():
    temple = request.args.get("temple", ""); ds = request.args.get("date", "")
    if temple not in TEMPLE_INFO:
        return jsonify(error="Unknown temple"), 400
    try:
        d = dt.date.fromisoformat(ds)
    except ValueError:
        return jsonify(error="Date must be YYYY-MM-DD"), 400
    level, probs = predict(temple, d)
    time_slot, wait, tip = ADVICE[level]
    week = []
    for i in range(7):
        di = d + dt.timedelta(days=i); li, _ = predict(temple, di)
        week.append(dict(date=di.isoformat(), day=di.strftime("%a"), level=li, festival=festival(di)))
    best = min(week, key=lambda w: LEVELS.index(w["level"]))
    c = calendar_features(d)
    return jsonify(temple=temple, date=d.isoformat(), level=level, probabilities=probs, best_time=time_slot,
                   waiting_time=wait, tip=tip, week=week, best_day=best, festival=c["festival"],
                   tamil_month=c["tamil_month"], flags={k: bool(c[k]) for k in
                   ("weekend", "public_holiday", "school_holiday", "long_weekend", "is_amavasai", "is_pournami", "is_ekadasi", "is_pradosham")},
                   info=TEMPLE_INFO[temple], maps="https://www.google.com/maps/search/?api=1&query=" +
                   urllib.parse.quote(temple + " Tamil Nadu"))


if __name__ == "__main__":
    app.run(debug=True)
