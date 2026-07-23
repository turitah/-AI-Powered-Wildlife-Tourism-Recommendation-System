import unittest

from app import app


class AppTests(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_predict_returns_success_response(self):
        response = self.client.post(
            "/predict",
            data={
                "animal": "Lion",
                "weather": "Sunny",
                "time_of_day": "Morning",
                "season": "Dry Season",
            },
            content_type="application/x-www-form-urlencoded",
        )

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["status"], "success")
        self.assertIn("recommendation", payload)


if __name__ == "__main__":
    unittest.main()
