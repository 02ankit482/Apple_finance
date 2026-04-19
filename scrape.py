import requests
import os

CIK = "0000320193"
BASE_URL = f"https://data.sec.gov/submissions/CIK{CIK}.json"

HEADERS = {
    "User-Agent": "AnkitYadav ankit.yadav@nablon.ai"
}


def get_10k_filings():
    data = requests.get(BASE_URL, headers=HEADERS).json()

    filings = data["filings"]["recent"]

    results = []

    for form, accession, doc in zip(
        filings["form"],
        filings["accessionNumber"],
        filings["primaryDocument"]
    ):
        if form == "10-K":
            accession_clean = accession.replace("-", "")
            url = f"https://www.sec.gov/Archives/edgar/data/{int(CIK)}/{accession_clean}/{doc}"
            results.append(url)

    return results


def download_html(url, idx):
    res = requests.get(url, headers=HEADERS)
    with open(f"filings/10k_{idx}.html", "w", encoding="utf-8") as f:
        f.write(res.text)


def main():
    os.makedirs("filings", exist_ok=True)

    links = get_10k_filings()

    print("Found 10-K filings:", links)

    for i, link in enumerate(links[:5]):
        download_html(link, i)


if __name__ == "__main__":
    main()