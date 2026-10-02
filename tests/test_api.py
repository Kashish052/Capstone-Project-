import unittest
from src.api import app


class ApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import setup_project

        setup_project.create_fixture()
        setup_project.ingest()
        setup_project.train_all()

    def setUp(self):
        self.client = app.test_client()

    def test_root(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json["status"], "running")

    def test_health(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json["status"], "healthy")

    def test_global_prediction(self):
        response = self.client.post("/predict?date=2019-12-01&duration=7")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json["country"], "ALL")
        self.assertEqual(len(response.json["daily_predictions"]), 7)

    def test_country_prediction(self):
        response = self.client.post(
            "/predict?date=2019-12-01&duration=7&country=Australia"
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json["country"], "Australia")

    def test_invalid_date_parameter(self):
        response = self.client.post("/predict?duration=7")
        self.assertEqual(response.status_code, 400)


if __name__ == "__main__":
    unittest.main()
