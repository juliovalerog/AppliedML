"""With Uvicorn running: python -m serving.client."""

# Decode the server's response and print its JSON fields readably.
import json
# Locate the complete customer body beside this module, independent of the terminal directory.
from pathlib import Path
# Make the HTTP request without installing an additional client library.
from urllib.request import Request, urlopen


def main():
    # Send the stored raw JSON bytes, keeping the same nineteen-field example as the lesson.
    payload = Path(__file__).with_name("customer.json").read_bytes()
    # Point to the local prediction route and declare that the body contains JSON.
    request = Request(
        "http://127.0.0.1:8000/predict", data=payload,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    # Send the request, limit waiting to ten seconds and close the response after reading it.
    with urlopen(request, timeout=10) as response:
        # Inspect the status before interpreting the body as a successful model prediction.
        print("HTTP", response.status)
        # Read probability and model label as JSON, then format them for classroom inspection.
        print(json.dumps(json.load(response), indent=2))


# Run a request only when invoked as a command, not when another module imports this file.
if __name__ == "__main__":
    main()
