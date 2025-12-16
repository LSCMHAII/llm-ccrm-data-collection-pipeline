import os
import json
import re
import time
import argparse
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

from utils import (
    linkExtractor, extract_year, get_html_content, parse_html,
    get_title, extract_press_release_content, get_date_time,
    create_json_data, save_json_data
)


def get_url_label(base_url):
    lang = "cn" if "/cn/" in base_url else "en"
    for label in ["speech", "press", "replies"]:
        if label in base_url:
            return f"{label}_{lang}"
    return f"unknown_{lang}"

def extract_pressrelease(base_path, base_url, year_start, year_end=None, month=None, get_links=True):
    year_end = year_start if year_end is None else year_end
    link_path = os.path.join(base_path, 'links')
    data_base_path = os.path.join(base_path, 'data')
    url_label = get_url_label(base_url)

    for year in range(year_start, year_end + 1):
        links_file = os.path.join(link_path, f"{year}_links_{url_label}.json")
        if get_links:
            linkExtractor(link_path, base_url, year, month)  # 傳入 month

        if not os.path.exists(links_file):
            print(f"Links file not found: {links_file}")
            continue

        with open(links_file, "r", encoding="utf-8") as f:
            links_data = json.load(f)

        data_path = os.path.join(data_base_path, f'{year}')
        os.makedirs(data_path, exist_ok=True)

        for link in links_data.get("links", []):
            print(link)

            year = extract_year(link)
            if year is not None:
                link = '/'.join(link.split('/')[1:])
                url = f"{base_url}/{link}"
                html_content = get_html_content(url)

                if html_content:
                    soup = parse_html(html_content)
                    title = get_title(soup)
                    pressrelease = extract_press_release_content(soup)

                    if pressrelease:
                        pattern = r'&amp;amp;lt;br/&amp;amp;gt;\n|&amp;amp;lt;/p&amp;amp;gt;\n&amp;amp;lt;p&amp;amp;gt;|&amp;amp;lt;br/&amp;amp;gt;\r\n'
                        content = re.split(pattern, str(pressrelease))

                        try:
                            date, time_str = get_date_time(content[-2])
                            data = create_json_data(title, date, time_str)
                            content = content[:-2]
                        except (IndexError, AttributeError):
                            print(f"Error with {link}")
                            continue

                        content_dict = {
                            f"p{i+1}": part.lstrip("&amp;amp;lt;br/&amp;amp;gt;\n").strip()
                            for i, part in enumerate(content) if part.strip()
                        }
                        data["content"] = content_dict

                        file_name = url.split("/")[-1] + ".json"
                        file_path = os.path.join(data_path, file_name)
                        save_json_data(data, file_path)
                    else:
                        print(f"Press release element not found for {url}")
                else:
                    print(f"Error fetching HTML content for {url}")
            else:
                print(f"Year not found in the URL: {link}")

def main_pressrelease(base_path, month=None):
    config_path = os.path.join(base_path, 'links_config.json')
    with open(config_path, 'r', encoding='utf-8') as f:
        config_data = json.load(f)

    year_start = 2025
    year_end = 2025
    get_links = True
    
    for category in config_data:
            urls = config_data.get(category, [])
            for base_url in urls:
                print(f"\n--- Processing {category} URL: {base_url} ---")
                extract_pressrelease(base_path, base_url, year_start, year_end, month, get_links)
                
# Download Legco Panel Paper links
def setup_driver():
    options = Options()
    options.add_argument('--headless')
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-gpu")

    driver_path = ChromeDriverManager().install()
    service = Service(driver_path)

    return webdriver.Chrome(service=service, options=options)

def fetch_page_source(url):
    driver = setup_driver()
    driver.get(url)
    time.sleep(5)
    page_source = driver.page_source
    driver.quit()
    return page_source

def extract_year_legco(soup, fallback_year="output"):
    content_div = soup.find('div', {'data-unit': 'content'})
    if content_div:
        h1_tag = content_div.find('h1')
        if h1_tag:
            match = re.search(r'(\d{4})$', h1_tag.text.strip())
            if match:
                return match.group(1)
    return fallback_year

def extract_links_legco(soup, base_link):
    urls = []
    wrapper_div = soup.find('div', {'data-unit': 'table-card-view-wrapper'})
    if wrapper_div:
        row_data_divs = wrapper_div.find_all('div', {'data-unit': 'row-data'})
        for row_div in row_data_divs:
            value_divs = row_div.find_all('div', {'data-unit': 'value'})
            for value_div in value_divs:
                for a_tag in value_div.find_all('a', href=True):
                    href = a_tag['href']
                    full_url = href if href.startswith("http") else base_link + href
                    urls.append(full_url)
    return urls

def save_links_to_json(links, year, lang, output_folder):
    filename = f"{year}_links_panel_paper_{lang}.json"
    output_path = f"{output_folder}/{filename}"
    data = {"links": links}
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return output_path

def main_legco(config, base_path='.'):
    link_path = os.path.join(base_path, 'links')

    for lang in ["chi", "eng"]:
        base_url = config[f"{lang}_link"]
        full_url = base_url + config["direct_path"]
        print(f"正在擷取 {lang} 版本的資料：{full_url}")

        html = fetch_page_source(full_url)
        soup = BeautifulSoup(html, 'html.parser')

        fallback_year = re.search(r'(\d{4})', config["direct_path"]).group(1)
        year = extract_year_legco(soup, fallback_year)

        links = extract_links_legco(soup, "https://www.legco.gov.hk")
        output_file = save_links_to_json(links, year, lang, link_path)

        print(f"[{lang.upper()}] 共擷取 {len(links)} 筆連結，已儲存至：{output_file}")

def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(data, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def get_links_difference(old_links, new_links):
    return list(set(new_links) - set(old_links))

def filter_links(links, prefix="archive/"):
    return [link for link in links if isinstance(link, str) and not link.startswith(prefix)]

def safe_load_links(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        links = data.get("links", [])
        return links if isinstance(links, list) else []
    except Exception as e:
        print(f"❌ Error loading {path}: {e}")
        return []

def main_link_filter(config_path):
    config_data = load_json(config_path)

    for category, paths in config_data.items():
        print(f"\n🔍 Dataset: {category}")

        chi_old = filter_links(safe_load_links(paths['chi_old']))
        eng_old = filter_links(safe_load_links(paths['eng_old']))
        chi_new = filter_links(safe_load_links(paths['chi_new']))
        eng_new = filter_links(safe_load_links(paths['eng_new']))

        chi_diff = get_links_difference(chi_old, chi_new)
        eng_diff = get_links_difference(eng_old, eng_new)

        save_json({"links": chi_diff}, f"{category}_chi_diff_filtered.json")
        save_json({"links": eng_diff}, f"{category}_eng_diff_filtered.json")
        
        save_json({"links": chi_new}, paths['chi_old'])
        save_json({"links": eng_new}, paths['eng_old'])

        print(f"✅ Saved {category} filtered links")
# ---------------- 主流程入口 ----------------
def main_combined(base_path=None):
    if base_path is None:
        base_path = '.'
    month = 12
    print("開始執行第一組代碼：新聞稿擷取")
    main_pressrelease(base_path, month)

    print("\n第一組代碼完成，開始執行第二組代碼：立法會連結擷取")
    legco_config_path = os.path.join(base_path, 'panel_paper_link_config.json')
    with open(legco_config_path, 'r', encoding='utf-8') as f:
        legco_config = json.load(f)
    main_legco(legco_config, base_path)

    print("\n第二組代碼完成，開始執行第三組代碼：比對新舊連結")
    link_filter_config_path = os.path.join(base_path, 'filtering_link_config.json')
    main_link_filter(link_filter_config_path)

if __name__ == '__main__':
    main_combined()

