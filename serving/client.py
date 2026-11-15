"""With Uvicorn running: python -m serving.client."""

import json
from pathlib import Path
from urllib.request import Request, urlopen


def main():
    payload = Path(__file__).with_name("customer.json").read_bytes()
    request = Request(
        "http://127.0.0.1:8000/predict", data=payload,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urlopen(request, timeout=10) as response:
        print("HTTP", response.status)
        print(json.dumps(json.load(response), indent=2))


if __name__ == "__main__":
    main()
