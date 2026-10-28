"""Shared feature engineering for TempleSense AI (used by data generation, training and the Flask app)."""
import datetime as dt
import pandas as pd

SYNODIC = 29.530588853
REF_NEW_MOON = dt.datetime(2000, 1, 6, 18, 14)

TAMIL_MONTHS = ["Chithirai", "Vaikasi", "Aani", "Aadi", "Aavani", "Purattasi",
                "Aippasi", "Karthigai", "Margazhi", "Thai", "Maasi", "Panguni"]
MONTH_STARTS = [(4, 14), (5, 15), (6, 15), (7, 17), (8, 17), (9, 17),
                (10, 18), (11, 17), (12, 16), (1, 14), (2, 13), (3, 14)]
FESTIVALS = ["None", "Pongal", "Tamil New Year", "Aadi Perukku", "Deepavali", "Vinayagar Chathurthi",
             "Chitra Pournami", "Thai Poosam", "Panguni Uthiram", "Karthigai Deepam",
             "Vaikunta Ekadasi", "Maha Shivaratri", "Skanda Sashti"]
WEATHER = ["Sunny", "Cloudy", "Rainy"]
SEASONS = ["Winter", "Summer", "Southwest Monsoon", "Northeast Monsoon"]
LEVELS = ["Low", "Moderate", "High", "Very High"]

# name, district, deity type, popularity (share of capacity on a normal day), special festivals
TEMPLES = [
    ("Meenakshi Amman Temple", "Madurai", "Amman", 0.55, ["Chitra Pournami", "Tamil New Year"]),
    ("Brihadeeswarar Temple", "Thanjavur", "Shiva", 0.40, ["Maha Shivaratri", "Pongal"]),
    ("Ramanathaswamy Temple", "Ramanathapuram", "Shiva", 0.50, ["Maha Shivaratri"]),
    ("Srirangam Ranganathaswamy Temple", "Tiruchirappalli", "Vishnu", 0.50, ["Vaikunta Ekadasi"]),
    ("Arunachaleswarar Temple", "Tiruvannamalai", "Shiva", 0.50, ["Karthigai Deepam", "Maha Shivaratri"]),
    ("Kapaleeshwarar Temple", "Chennai", "Shiva", 0.40, ["Panguni Uthiram", "Maha Shivaratri"]),
    ("Parthasarathy Temple", "Chennai", "Vishnu", 0.32, ["Vaikunta Ekadasi"]),
    ("Palani Murugan Temple", "Dindigul", "Murugan", 0.58, ["Thai Poosam", "Panguni Uthiram"]),
    ("Thiruchendur Murugan Temple", "Thoothukudi", "Murugan", 0.45, ["Skanda Sashti"]),
    ("Swamimalai Murugan Temple", "Thanjavur", "Murugan", 0.28, ["Thai Poosam"]),
    ("Ekambareswarar Temple", "Kanchipuram", "Shiva", 0.36, ["Panguni Uthiram", "Maha Shivaratri"]),
    ("Kamakshi Amman Temple", "Kanchipuram", "Amman", 0.40, ["Tamil New Year"]),
    ("Samayapuram Mariamman Temple", "Tiruchirappalli", "Amman", 0.50, ["Chitra Pournami", "Aadi Perukku"]),
    ("Jambukeswarar Temple", "Tiruchirappalli", "Shiva", 0.30, ["Maha Shivaratri"]),
    ("Thillai Nataraja Temple", "Cuddalore", "Shiva", 0.34, ["Maha Shivaratri"]),
    ("Kanyakumari Bhagavathi Amman Temple", "Kanyakumari", "Amman", 0.42, ["Chitra Pournami"]),
    ("Suchindram Thanumalayan Temple", "Kanyakumari", "Shiva", 0.26, ["Maha Shivaratri"]),
    ("Thirupparankundram Murugan Temple", "Madurai", "Murugan", 0.38, ["Skanda Sashti", "Thai Poosam"]),
    ("Vadapalani Murugan Temple", "Chennai", "Murugan", 0.42, ["Thai Poosam", "Skanda Sashti"]),
    ("Rockfort Ucchi Pillayar Temple", "Tiruchirappalli", "Ganesha", 0.30, ["Vinayagar Chathurthi"]),
]
TEMPLE_NAMES = [t[0] for t in TEMPLES]
DISTRICTS = sorted({t[1] for t in TEMPLES})
TEMPLE_INFO = {t[0]: dict(district=t[1], deity=t[2], pop=t[3], special=t[4]) for t in TEMPLES}

DEEPAVALI = {2022: (10, 24), 2023: (11, 12), 2024: (11, 1), 2025: (10, 20), 2026: (11, 8), 2027: (10, 29)}
VINAYAGAR = {2022: (8, 31), 2023: (9, 18), 2024: (9, 7), 2025: (8, 27), 2026: (9, 14), 2027: (9, 4)}
FIXED_HOLIDAYS = {(1, 1), (1, 15), (1, 16), (1, 26), (4, 14), (5, 1), (8, 15), (10, 2), (12, 25)}
MONTH_TEMP = [25, 26.5, 29, 31, 32.5, 31, 30, 29.5, 29, 27.5, 26, 25]
MONTH_RAIN_P = [.05, .04, .05, .10, .12, .15, .17, .20, .22, .40, .50, .30]


def moon_age(d):
    delta = dt.datetime(d.year, d.month, d.day, 12) - REF_NEW_MOON
    return (delta.total_seconds() / 86400.0) % SYNODIC


def tithi(d):
    """Approximate lunar day 1..30 (15 = Pournami, 30 = Amavasai)."""
    return min(int(moon_age(d) / (SYNODIC / 30)) + 1, 30)


def tamil_month(d):
    key = (d.month, d.day)
    best = None
    for i, s in enumerate(MONTH_STARTS):
        if s <= key and (best is None or s > MONTH_STARTS[best]):
            best = i
    return TAMIL_MONTHS[8 if best is None else best]


def season(d):
    m = d.month
    if m in (1, 2): return "Winter"
    if m in (3, 4, 5): return "Summer"
    if m in (6, 7, 8, 9): return "Southwest Monsoon"
    return "Northeast Monsoon"


def festival(d):
    t, tm = tithi(d), tamil_month(d)
    key = (d.month, d.day)
    if key == (1, 15): return "Pongal"
    if key == (4, 14): return "Tamil New Year"
    if key == (8, 3): return "Aadi Perukku"
    if DEEPAVALI.get(d.year) == key: return "Deepavali"
    if VINAYAGAR.get(d.year) == key: return "Vinayagar Chathurthi"
    if t == 15 and tm in ("Chithirai", "Thai", "Panguni", "Karthigai"):
        return {"Chithirai": "Chitra Pournami", "Thai": "Thai Poosam",
                "Panguni": "Panguni Uthiram", "Karthigai": "Karthigai Deepam"}[tm]
    if t == 11 and tm == "Margazhi": return "Vaikunta Ekadasi"
    if t == 29 and tm == "Maasi": return "Maha Shivaratri"
    if t == 6 and tm == "Aippasi": return "Skanda Sashti"
    return "None"


def is_public_holiday(d):
    return int((d.month, d.day) in FIXED_HOLIDAYS or festival(d) in
               ("Pongal", "Tamil New Year", "Deepavali", "Vinayagar Chathurthi"))


def is_school_holiday(d):
    return int(d.month == 5 or (d.month == 12 and d.day >= 23) or (d.month == 1 and d.day <= 2)
               or (d.month == 10 and d.day <= 8))


def is_long_weekend(d):
    w = d.weekday()
    if w not in (4, 5, 6, 0):
        return 0
    fri = d - dt.timedelta(days=(w - 4) % 7)
    mon = fri + dt.timedelta(days=3)
    return int(is_public_holiday(fri) or is_public_holiday(mon))


def festival_proximity(d):
    if festival(d) != "None":
        return 0
    return int(any(festival(d + dt.timedelta(days=o)) != "None" for o in (-2, -1, 1, 2)))


def calendar_features(d):
    t = tithi(d)
    return dict(
        day_of_week=d.weekday(), weekend=int(d.weekday() >= 5), month=d.month,
        tamil_month=tamil_month(d), season=season(d), festival=festival(d),
        is_amavasai=int(t == 30), is_pournami=int(t == 15),
        is_ekadasi=int(t in (11, 26)), is_pradosham=int(t in (13, 28)),
        festival_proximity=festival_proximity(d), public_holiday=is_public_holiday(d),
        school_holiday=is_school_holiday(d), long_weekend=is_long_weekend(d))


def expected_weather(d):
    p = MONTH_RAIN_P[d.month - 1]
    w = "Rainy" if p >= 0.35 else "Sunny"
    return dict(weather=w, temperature=MONTH_TEMP[d.month - 1] - (2 if w == "Rainy" else 0),
                rainfall_mm=12.0 if w == "Rainy" else 0.0)


NUMERIC = ["day_of_week", "weekend", "month", "is_amavasai", "is_pournami", "is_ekadasi", "is_pradosham",
           "festival_proximity", "public_holiday", "school_holiday", "long_weekend", "temperature", "rainfall_mm"]
CATEGORICAL = {"temple": TEMPLE_NAMES, "district": DISTRICTS, "tamil_month": TAMIL_MONTHS,
               "festival": FESTIVALS, "weather": WEATHER, "season": SEASONS}
FEATURES = NUMERIC + list(CATEGORICAL)


def encode(df):
    out = df[NUMERIC].copy()
    for col, cats in CATEGORICAL.items():
        out[col] = df[col].map({c: i for i, c in enumerate(cats)}).astype(int)
    return out[FEATURES]


def build_row(temple, d, weather=None):
    row = calendar_features(d)
    row.update(expected_weather(d))
    if weather:
        row["weather"] = weather
    row.update(temple=temple, district=TEMPLE_INFO[temple]["district"])
    return pd.DataFrame([row])
