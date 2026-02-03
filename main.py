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
from datetime import datetime
from utils import (
    linkExtractor, extract_year, get_html_content, parse_html,
    get_title, extract_press_release_content, get_date_time,
    create_json_data, save_json_data
)
# Download Press Relase, Speech and Legco Qna links
def get_url_label(base_url):
    # Check URL Language (Chi or Eng)
    lang = "cn" if "/cn/" in base_url else "en"
    # Check the type 
    for label in ["speech", "press", "replies"]:
        if label in base_url:
            return f"{label}_{lang}"
    return f"unknown_{lang}"

# Extract Press Release, Speech and Legco Qna content and Save to JSON 
def extract_pressrelease(base_path, base_url, year_start, year_end=None, month=None, get_links=True):
    year_end = year_start if year_end is None else year_end
    link_path = os.path.join(base_path, 'links')
    data_base_path = os.path.join(base_path, 'data')
    url_label = get_url_label(base_url)

    today = datetime.now()
    curr_year = today.year
    curr_month = today.month
    is_first_day_of_month = (today.day == 1) and (month is None)

    def get_prev_year_month(y, m):
        if m == 1:
            return (y - 1, 12)
        return (y, m - 1)
    prev_year, prev_month = get_prev_year_month(curr_year, curr_month)
    # Relative joins processed in this round (to avoid duplication)
    processed_links = set()

    # Process Links Data
    def process_links_data(links_data, target_year):
        data_path = os.path.join(data_base_path, f'{target_year}')
        os.makedirs(data_path, exist_ok=True)

        for link in links_data.get("links", []):
            # Deduplication: In the same round of execution, identical relative links are not processed repeatedly.
            if link in processed_links:
                continue
            processed_links.add(link)
            print(f"Processing links: {link}")
            link_year = extract_year(link)
            if link_year is not None:
                # Combine the URL
                link_trimmed = '/'.join(link.split('/')[1:])
                url = f"{base_url}/{link_trimmed}"
                html_content = get_html_content(url)

                if html_content:
                    soup = parse_html(html_content)
                    title = get_title(soup)
                    pressrelease = extract_press_release_content(soup)
                    if pressrelease:
                        # Preserve your original segmentation and decoding logic
                        pattern = r'&amp;amp;amp;amp;amp;lt;br/&amp;amp;amp;amp;amp;gt;\n|&amp;amp;amp;amp;amp;lt;/p&amp;amp;amp;amp;amp;gt;\n&amp;amp;amp;amp;amp;lt;p&amp;amp;amp;amp;amp;gt;|&amp;amp;amp;amp;amp;lt;br/&amp;amp;amp;amp;amp;gt;\r\n'
                        content = re.split(pattern, str(pressrelease))
                        try:
                            date, time_str = get_date_time(content[-2])
                            data = create_json_data(title, date, time_str)
                            content = content[:-2]
                        except (IndexError, AttributeError):
                            print(f"❌ Date parsing failed：{link}")
                            continue
                        content_dict = {f"p{i+1}": part.lstrip("&amp;amp;amp;amp;amp;lt;br/&amp;amp;amp;amp;amp;gt;\n").strip() for i, part in enumerate(content) if part.strip()}
                        data["content"] = content_dict
                        file_name = url.split("/")[-1] + ".json"
                        file_path = os.path.join(data_path, file_name)
                        save_json_data(data, file_path)


                    else:
                        print(f"⚠️ No press release content elements found：{url}")
                else:
                    print(f"⚠️ HTML fetching failed：{url}")
            else:
                print(f"⚠️ does not include year：{link}")

    for y in range(year_start, year_end + 1):
        links_file_this_year = os.path.join(link_path, f"{y}_links_{url_label}.json")

        # Step 1: Dealing with "Today" and "Monthly Fallback"
        day_links_data = {}
        month_links_data = {}

        if get_links:
            # Fetch『Today』links
            print(f"➡️ First retrieval: Based on 'today' (month=None)")
            linkExtractor(link_path, base_url, y, month=None)

            # Read today's archives (which have not yet been overwritten by last month's).
            if os.path.exists(links_file_this_year):
                with open(links_file_this_year, "r", encoding="utf-8") as f:
                    day_links_data = json.load(f)
                if day_links_data.get("links"):
                    print(f"✅ Found {len(day_links_data.get('links', []))} links today and will process them first")
                else:
                    print("⚠️ No links today（links empty），will fallback to fetch the entire month")
            else:
                print(f"⚠️ The linked archive for today does not exist; will fallback to retrieve the entire month...")

        # Determine if a fallback is needed to fetch the entire month.
        need_month_fallback = (
            not day_links_data.get("links", [])
        )

        if need_month_fallback:
            # If no month is specified, the default value is "the month of today"
            fallback_month = month if month is not None else today.month
            print(f"➡️ Fallback: Shift to fetch the entire month. (month={fallback_month})")
            linkExtractor(link_path, base_url, y, month=fallback_month)

            if os.path.exists(links_file_this_year):
                with open(links_file_this_year, "r", encoding="utf-8") as f:
                    month_links_data = json.load(f)

            if not month_links_data.get("links"):
                print(f"❌ Entire month (month={fallback_month}) no links，skipping the year {y} for current-month processing")
            else:
                # Process links for the entire month (or today) first.
                process_links_data(month_links_data, y)
        else:
            # If you find today's link, process today's request first
            process_links_data(day_links_data, y)

        # Step 2 : If it's the 1st of each month, process the 'previous month' separately
        if is_first_day_of_month:
            # This call was made just a month ago; to avoid overwriting "today's" files.
            print(f"🗓️ Detected new month (day=1). Also fetching previous month {prev_year}-{prev_month:02d} links...")
            linkExtractor(link_path, base_url, prev_year, month=prev_month)

            if prev_year == y:
                # The previous month of the same year: Read directly from the "current year" file (which was just overwritten by the previous month).
                if os.path.exists(links_file_this_year):
                    with open(links_file_this_year, "r", encoding="utf-8") as f:
                        prev_month_links_data = json.load(f)
                    if prev_month_links_data.get("links"):
                        print(f"🟡 Additionally processing previous month ({prev_year}-{prev_month:02d}) links in the same year.")
                        process_links_data(prev_month_links_data, y)
                    else:
                        print(f"ℹ️ Previous month ({prev_year}-{prev_month:02d}) has no links in file {links_file_this_year}")
                else:
                    print(f"ℹ️ Previous month file not found for same year: {links_file_this_year}")
            else:
                # One month into the new year: Retrieving files from the previous year.
                prev_links_file = os.path.join(link_path, f"{prev_year}_links_{url_label}.json")
                if not os.path.exists(prev_links_file):
                    print(f"ℹ️ Previous month ({prev_year}-{prev_month:02d}) file not found: {prev_links_file}")
                else:
                    with open(prev_links_file, "r", encoding="utf-8") as f:
                        prev_month_links_data = json.load(f)
                    if prev_month_links_data.get("links"):
                        print(f"🟡 Additionally processing previous month ({prev_year}-{prev_month:02d}) links across year {prev_year}.")
                        process_links_data(prev_month_links_data, prev_year)
                    else:
                        print(f"ℹ️ Previous month ({prev_year}-{prev_month:02d}) has no links in file {prev_links_file}")

def main_pressrelease(base_path, month=None):
    config_path = os.path.join(base_path, 'links_config.json')
    with open(config_path, 'r', encoding='utf-8') as f:
        config_data = json.load(f)

    # current setting (fixed for 2026); if you want it to be dynamic, you can change it to datetime.now().year.
    year_start = 2026
    year_end = datetime.now().year
    get_links = True

    print(f"➡️ Execution strategy：First, capture instances where there are no links for 『today』 → automatically fallback to capture the 『entire month』")
    if month is not None:
        print(f"📅 Specified month：{month}（fallback use this month; if not specified, it will use today's month.）")

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
    config["direct_path"] = datetime.now().strftime("%Y%m%d")

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

def append_to_existing_file(new_links, file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            old_data = json.load(f)
            old_links = old_data.get("links", [])
    except FileNotFoundError:
        old_links = []

    # Merge and sort in reverse order
    combined_links = sorted(set(old_links + new_links), reverse=True)

    # Write back to the original link file
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump({"links": combined_links}, f, indent=2, ensure_ascii=False)

def main_link_filter(config_path):
    config_data = load_json(config_path)

    for category, paths in config_data.items():
        print(f"\n🔍 Dataset: {category}")

        chi_old_links = filter_links(safe_load_links(paths['chi_old']))
        eng_old_links = filter_links(safe_load_links(paths['eng_old']))
        chi_new_links = filter_links(safe_load_links(paths['chi_new']))
        eng_new_links = filter_links(safe_load_links(paths['eng_new']))

        chi_diff = get_links_difference(chi_old_links, chi_new_links)
        eng_diff = get_links_difference(eng_old_links, eng_new_links)

        # 將新data link添加至舊的link檔案
        append_to_existing_file(chi_diff, paths['chi_old'])
        append_to_existing_file(eng_diff, paths['eng_old'])

        print(f"✅ Updated {category}: appended and sorted links in old files")

# Entry Point
def main_combined(base_path=None, month=None):
    if base_path is None:
        base_path = '.'

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
    parser = argparse.ArgumentParser(description="新聞稿與立法會資料擷取工具")
    parser.add_argument('--month', type=int, help="指定月份 (例如: 12)")
    args = parser.parse_args()
    main_combined(month=args.month)
