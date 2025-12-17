import requests
from bs4 import BeautifulSoup
import json, os, re
from datetime import datetime

def extract_year(url):
    match = re.search(r'\d{4}', url)
    return int(match.group()) if match else None

def get_html_content(url):
    response = requests.get(url)
    if response.status_code == 200:
        return response.content
    print(f"Error: {response.status_code}")
    return None

def parse_html(html_content):
    return BeautifulSoup(html_content, "html.parser")

def get_title(soup):
    title_element = soup.find("h2")
    return title_element.get_text(strip=True) if title_element else ""

def get_description(soup):
    elements = soup.find_all("h2")
    return elements[1].get_text(strip=True) if len(elements) > 1 else ""

def get_date_time(content):
    pattern = r'Ends/(\w+,\s*\w+\s+\d+,\s*\d+)'
    if isinstance(content, str):
        match = re.search(pattern, content)
        return (match.group(1), "") if match else ("", "")
    elif isinstance(content, list):
        for item in content:
            if isinstance(item, str):
                match = re.search(pattern, item)
                if match:
                    return (match.group(1), "")
    return "", ""

def extract_press_release_content(soup):
    content = soup.find("div", class_="contentThin")
    if content:
        elements = content.find_all("p")
        return "\n".join(str(e) for e in elements) if elements else ""
    return ""

def create_json_data(title, date, time):
    return {
        "metadata": {"date": date, "time": time},
        "title": title,
        "content": []
    }

def save_json_data(data, filename):
    with open(filename, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=4)
    print("JSON file created successfully.")



def linkExtractor(data_path, base_url, year, month=None):
    # 判斷語言與標籤
    lang = "cn" if "/cn/" in base_url else "en"
    for label in ["speech", "press", "replies"]:
        if label in base_url:
            url_label = f"{label}_{lang}"
            break
    else:
        url_label = f"unknown_{lang}"

    # 當前日期
    today = datetime.now()
    today_day = today.day
    today_month = today.month
    today_year = today.year

    url = f"{base_url}/{year}.html"
    response = requests.get(url)

    if response.status_code == 200:
        soup = BeautifulSoup(response.content, "html.parser")
        links = []

        # 搵table中的所有行
        rows = soup.find_all("tr")
        for row in rows:
            cols = row.find_all("td")
            if len(cols) >= 2:
                date_text = cols[0].get_text(strip=True)  # 例如 "09/12/2025"
                link_tag = cols[1].find("a", href=True)
                if link_tag:
                    href = link_tag["href"]
                    try:
                        day, mth, yr = map(int, date_text.split("/"))
                        # 檢查月份
                        if month and mth != month:
                            continue
                        # 檢查是否是今天日期
                        if day != today_day or mth != today_month or yr != today_year:
                            continue
                    except ValueError:
                        continue
                    links.append(href)

        # JSON 資料
        links_json = {
            # "extracted_date": today.strftime("%Y-%m-%d"),
            "links": links
        }

        filename = f"{year}_links_{url_label}.json"
        os.makedirs(data_path, exist_ok=True)
        json_path = os.path.join(data_path, filename)

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(links_json, f, ensure_ascii=False, indent=4)

        print(f"Saved {json_path} (共 {len(links)} 筆連結, 僅保留今天日期: {today.strftime('%d/%m/%Y')})")
        return filename
