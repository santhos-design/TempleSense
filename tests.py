"""Unit + API tests. Run:  python -m unittest tests -v"""
import datetime as dt, unittest
from features import *
from app import app, predict


class CalendarTests(unittest.TestCase):
    def test_thai_poosam_2024(self): self.assertEqual(festival(dt.date(2024, 1, 25)), "Thai Poosam")
    def test_chitra_pournami_2024(self): self.assertEqual(festival(dt.date(2024, 4, 23)), "Chitra Pournami")
    def test_maha_shivaratri_2024(self): self.assertEqual(festival(dt.date(2024, 3, 8)), "Maha Shivaratri")
    def test_vaikunta_ekadasi_2025(self): self.assertEqual(festival(dt.date(2025, 12, 30)), "Vaikunta Ekadasi")
    def test_tamil_new_year_holiday(self): self.assertEqual(is_public_holiday(dt.date(2026, 4, 14)), 1)
    def test_weekend_flag(self): self.assertEqual(calendar_features(dt.date(2026, 10, 24))["weekend"], 1)
    def test_tamil_month(self): self.assertEqual(tamil_month(dt.date(2026, 1, 20)), "Thai")
    def test_amavasai_and_pournami_differ(self):
        d = dt.date(2024, 4, 8); self.assertEqual((is_amavasai := calendar_features(d)["is_amavasai"]), 1)
        self.assertEqual(calendar_features(dt.date(2024, 4, 23))["is_pournami"], 1)


class ModelTests(unittest.TestCase):
    def test_valid_level_for_every_temple(self):
        for t in TEMPLE_NAMES: self.assertIn(predict(t, dt.date(2026, 11, 7))[0], LEVELS)
    def test_probabilities_sum_to_one(self):
        self.assertAlmostEqual(sum(predict(TEMPLE_NAMES[0], dt.date(2026, 12, 1))[1].values()), 1.0, places=6)
    def test_festival_not_lower_than_ordinary_day(self):
        t = "Palani Murugan Temple"
        self.assertGreaterEqual(LEVELS.index(predict(t, dt.date(2024, 1, 25))[0]),
                                LEVELS.index(predict(t, dt.date(2024, 1, 31))[0]))


class ApiTests(unittest.TestCase):
    def setUp(self): self.c = app.test_client()
    def test_home(self): self.assertEqual(self.c.get("/").status_code, 200)
    def test_search_filter(self):
        r = self.c.get("/api/temples?district=Chennai").json
        self.assertTrue(r and all(x["district"] == "Chennai" for x in r))
    def test_predict_ok(self):
        r = self.c.get("/api/predict?temple=Meenakshi%20Amman%20Temple&date=2026-11-08")
        self.assertEqual(r.status_code, 200); self.assertIn(r.json["level"], LEVELS); self.assertEqual(len(r.json["week"]), 7)
    def test_unknown_temple(self): self.assertEqual(self.c.get("/api/predict?temple=x&date=2026-11-08").status_code, 400)
    def test_bad_date(self): self.assertEqual(self.c.get("/api/predict?temple=Palani%20Murugan%20Temple&date=abc").status_code, 400)


if __name__ == "__main__":
    unittest.main()
