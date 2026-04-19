import os
import re
from bs4 import BeautifulSoup

INPUT_DIR = "filings"
OUTPUT_DIR = "processed"
SECTION_DIR = "sections"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(SECTION_DIR, exist_ok=True)

files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".html")]

for file in files:
    input_path = os.path.join(INPUT_DIR, file)

    with open(input_path, "r", encoding="utf-8") as f:
        html = f.read()

    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["script", "style"]):
        tag.extract()

    text = soup.get_text(separator="\n")

    lines = [line.strip() for line in text.splitlines()]
    text = "\n".join([line for line in lines if line])

    base_name = file.replace(".html", "")

    with open(os.path.join(OUTPUT_DIR, base_name + ".txt"), "w", encoding="utf-8") as f:
        f.write(text)

    lower_text = text.lower()


patterns = {
    "item_1": r"item\s+1\..*?(?=item\s+1a\.)",
    "item_1a": r"item\s+1a\..*?(?=item\s+7\.)",
    "item_7": r"item\s+7\..*?(?=item\s+7a\.)",
    "item_7a": r"item\s+7a\..*?(?=item\s+8\.)"
}

for key, pattern in patterns.items():
    matches = re.findall(pattern, lower_text, re.DOTALL)

    if matches:
        section_text = max(matches, key=len)  

        filename = f"{base_name}_{key}.txt"
        path = os.path.join(SECTION_DIR, filename)

        with open(path, "w", encoding="utf-8") as f:
            f.write(section_text)

    print(f"Processed: {file}")