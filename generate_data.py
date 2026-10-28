"""Generates the synthetic Tamil Nadu temple crowd datasets (formula-based, seeded for reproducibility)."""
import datetime as dt
import numpy as np, pandas as pd
from features import *

rng = np.random.default_rng(42)
FEST_MULT = {"Pongal": 1.9, "Tamil New Year": 2.1, "Aadi Perukku": 1.7, "Deepavali": 1.6,
             "Vinayagar Chathurthi": 1.6, "Chitra Pournami": 2.2, "Thai Poosam": 2.3,
             "Panguni Uthiram": 2.1, "Karthigai Deepam": 2.2, "Vaikunta Ekadasi": 2.3,
             "Maha Shivaratri": 2.2, "Skanda Sashti": 2.0}
WAIT = {"Low": (5, 20), "Moderate": (25, 50), "High": (55, 100), "Very High": (110, 220)}


def level(r):
    return "Low" if r < 0.42 else "Moderate" if r < 0.55 else "High" if r < 0.78 else "Very High"


rows = []
start, end = dt.date(2022, 1, 1), dt.date(2026, 9, 30)
days = [start + dt.timedelta(days=i) for i in range((end - start).days + 1)]
for name in TEMPLE_NAMES:
    info = TEMPLE_INFO[name]
    for d in days:
        c = calendar_features(d)
        p = MONTH_RAIN_P[d.month - 1]
        rainy = rng.random() < p
        weather = "Rainy" if rainy else ("Cloudy" if rng.random() < 0.3 else "Sunny")
        temp = round(MONTH_TEMP[d.month - 1] + rng.normal(0, 1.5) - (2 if rainy else 0), 1)
        rain = round(float(rng.exponential(12)), 1) if rainy else 0.0
        m = info["pop"]
        m *= 1.45 if c["weekend"] else 1.0
        m *= 1.35 if c["public_holiday"] else 1.0
        m *= 1.25 if c["long_weekend"] else 1.0
        m *= 1.20 if c["school_holiday"] else 1.0
        if c["festival"] != "None":
            m *= FEST_MULT[c["festival"]] * (1.5 if c["festival"] in info["special"] else 1.0) ** 1
        m *= 1.12 if c["festival_proximity"] else 1.0
        m *= 1.25 if c["is_pournami"] else 1.0
        m *= 1.20 if (c["is_amavasai"] and info["deity"] in ("Amman", "Shiva")) else 1.0
        m *= 1.20 if (c["is_pradosham"] and info["deity"] == "Shiva") else 1.0
        m *= 1.20 if (c["is_ekadasi"] and info["deity"] == "Vishnu") else 1.0
        m *= 0.80 if weather == "Rainy" else 1.0
        m *= 0.90 if temp > 34 else 1.0
        ratio = m * rng.lognormal(0, 0.10)
        lv = level(ratio)
        lo, hi = WAIT[lv]
        rows.append(dict(date=d, temple=name, district=info["district"], **c, weather=weather,
                         temperature=temp, rainfall_mm=rain, crowd_ratio=round(ratio, 3),
                         crowd_level=lv, waiting_time_min=int(rng.integers(lo, hi + 1))))

df = pd.DataFrame(rows)
df.to_csv("data/crowd_data.csv", index=False)
pd.DataFrame([dict(temple=t[0], district=t[1], deity=t[2], popularity=t[3]) for t in TEMPLES]).to_csv("data/temples.csv", index=False)
fe = [dict(date=d, festival=festival(d), tamil_month=tamil_month(d)) for d in days if festival(d) != "None"]
pd.DataFrame(fe).to_csv("data/festival_calendar.csv", index=False)
pd.DataFrame([dict(date=d, public_holiday=is_public_holiday(d), school_holiday=is_school_holiday(d),
                   long_weekend=is_long_weekend(d)) for d in days if is_public_holiday(d) or is_school_holiday(d)]
             ).to_csv("data/holidays.csv", index=False)
print(len(df), "rows"); print(df.crowd_level.value_counts(normalize=True).round(3))
