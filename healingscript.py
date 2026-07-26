import requests
import subprocess
import time

PROM_URL = "http://localhost:9090/api/v1/query"

QUERY = '''
100 * sum(rate(http_requests_total{status=~"[45].."}[5m]))
/
sum(rate(http_requests_total[5m]))
'''

THRESHOLD = 30

while True:
    try:
        response = requests.get(
            PROM_URL,
            params={"query": QUERY}
        )

        data = response.json()

        result = data["data"]["result"]

        if len(result) == 0:
            print("No error data yet")
        else:
            error_rate = float(result[0]["value"][1])

            print(f"Error Rate = {error_rate:.2f}%")

            if error_rate > THRESHOLD:
                print("ERROR RATE TOO HIGH!")
                print("Restarting Flask App...")

                subprocess.Popen(["python", "shopping.py"])

                time.sleep(10)

    except Exception as e:
        print("Error:", e)

    time.sleep(10)