import requests
import json
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter

url = "https://datasets-server.huggingface.co/first-rows"
params = {
    "dataset": "netop/Beam-Level-Traffic-Timeseries-Dataset",
    "config": "DLPRB",
    "split": "train_0w_5w",
}

def create_retry_session(total_retries: int = 5, backoff_factor: float = 0.8) -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=total_retries,
        backoff_factor=backoff_factor,
        status_forcelist=[500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session

def fetch_first_rows() -> dict:
    session = create_retry_session()
    # Add a User-Agent to avoid being blocked by generic bots
    headers = {"User-Agent": "ADM-script/1.0 (+https://huggingface.co)"}
    response = session.get(url, params=params, headers=headers, timeout=60)
    if response.status_code >= 400:
        # Try to provide actionable info on 5xx/4xx
        try:
            details = response.json()
        except Exception:
            details = {"text": response.text[:500]}
        raise requests.HTTPError(
            f"Failed to fetch first rows: HTTP {response.status_code}", response=response
        ) from None
    return response.json()

def list_splits_for_debug() -> None:
    splits_url = "https://datasets-server.huggingface.co/splits"
    query = {"dataset": params["dataset"]}
    try:
        r = requests.get(splits_url, params=query, timeout=30)
        if r.ok:
            info = r.json()
            # Print available configs/splits to help debug bad combo
            print("Available configs and splits:")
            for cfg in info.get("splits", []):
                cfg_name = cfg.get("config", "<unknown>")
                split_name = cfg.get("split", "<unknown>")
                print(f"- config={cfg_name} split={split_name}")
        else:
            print(f"Could not retrieve splits (HTTP {r.status_code}).")
    except Exception as e:
        print(f"Error fetching splits info: {e}")

try:
    data = fetch_first_rows()
    print(json.dumps(data, indent=2))
    for row in data.get("rows", [])[:5]:
        print(f"idx={row.get('row_idx')}: {row.get('row')}")
except requests.HTTPError as e:
    status = getattr(getattr(e, "response", None), "status_code", "<no-status>")
    print(f"Request failed with HTTP status {status}. Retried with backoff and still failed.")
    # Provide hints by showing available splits in case of bad config/split combo
    list_splits_for_debug()
except Exception as e:
    print(f"Unexpected error: {e}")