import os
import json
import requests
import pdfplumber
import markdownify
import re
import pandas as pd
import glob
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from urllib.parse import urlparse
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.config.parser import ConfigParser
from marker.output import text_from_rendered
from openpyxl import load_workbook

# === Extract Press Rlease, Speech, Legco QnA ===
def fetch_url_content(url):
    response = requests.get(url)
    response.encoding = 'utf-8'
    return response.text

def extract_specific_content_v1(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    headline = soup.find(id="PRHeadlineSpan")
    press_release = soup.find(id="pressrelease")
    headline_text = headline.get_text(separator='\n') if headline else None

    if press_release:
        for a in press_release.find_all('a'):
            a.attrs = {key: value for key, value in a.attrs.items() if key in ['href', 'style']}
        press_release_text = str(press_release)
        press_release_text_plain = press_release.get_text(separator='\n')
    else:
        press_release_text = None
        press_release_text_plain = None

    return headline_text, press_release_text, press_release_text_plain

def extract_specific_content_v2(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    content_div = soup.find(id="content")
    headline = content_div.find('h2') if content_div else None
    press_release = content_div.find_all('p') if content_div else None
    headline_text = headline.get_text(separator='\n\n') if headline else 'No Headline'

    if press_release:
        press_release_text = '\n\n'.join([str(p) for p in press_release])
        press_release_text_plain = '\n\n'.join([p.get_text(separator='\n') for p in press_release])
    else:
        press_release_text = 'No Press Release'
        press_release_text_plain = 'No Press Release'

    return headline_text, press_release_text, press_release_text_plain

def extract_metadata_chi(press_release_text_plain):
    date = ''
    time = ''
    date_match = re.search(r'(\d{4}年\d{1,2}月\d{1,2}日（星期[一二三四五六日]）)', press_release_text_plain)
    if date_match:
        date_str = date_match.group(1)
        week_map = {
            '一': 'Monday', '二': 'Tuesday', '三': 'Wednesday',
            '四': 'Thursday', '五': 'Friday', '六': 'Saturday', '日': 'Sunday'
        }
        for cn_week, en_week in week_map.items():
            date_str = date_str.replace(f'星期{cn_week}', en_week)
        date_str = date_str.replace('（', '(').replace('）', ')')
        date_obj = datetime.strptime(date_str, '%Y年%m月%d日(%A)')
        date = date_obj.strftime('%Y-%m-%d')

    time_match = re.search(r'香港時間(\d{1,2}時\d{1,2}分)', press_release_text_plain)
    if time_match:
        time_str = time_match.group(1)
        time_obj = datetime.strptime(time_str, '%H時%M分')
        time = time_obj.strftime('HKT %H:%M')

    return date, time

def extract_metadata_eng(press_release_text_plain):
    date = ''
    time = ''
    if "Ends/" in press_release_text_plain and "Issued at" in press_release_text_plain:
        date_raw = press_release_text_plain.split("Ends/")[-1].split("Issued at")[0].strip()
        match = re.search(r'([A-Za-z]+, [A-Za-z]+ \d{1,2}, \d{4})', date_raw)
        if match:
            try:
                date_obj = datetime.strptime(match.group(1), '%A, %B %d, %Y')
                date = date_obj.strftime('%Y-%m-%d')
            except Exception:
                date = date_raw
        else:
            date = date_raw
    if "Issued at" in press_release_text_plain and "NNNN" in press_release_text_plain:
        time = press_release_text_plain.split("Issued at")[-1].split("NNNN")[0].strip()
    return date, time

def convert_to_markdown(headline, press_release):
    markdown_headline = f"# {headline}\n\n"
    markdown_press_release = markdownify.markdownify(press_release, heading_style="ATX", strip=['a'])
    return markdown_headline + markdown_press_release

def save_as_markdown(content, filepath):
    with open(filepath, 'w', encoding='utf-8') as file:
        file.write(content)

def get_filename_from_url(url):
    base_name = os.path.basename(urlparse(url).path)
    return base_name if base_name.endswith('.md') else base_name + '.md'

def convert_markdown_to_json(markdown_content, date, time, url):
    lines = markdown_content.split('\n')
    metadata = {
        "date": date,
        "start_time": "",
        "end_time": "",
        "time": time,
        "is_ocr": False,
        "source_link": url
    }
    content = {}
    title = ""
    paragraph_count = 1

    for line in lines:
        if line.startswith("# "):
            title = line[2:].strip()
        elif line.strip():
            clean_line = line.strip().replace('\\', '').replace('"', '')
            content[f"p{paragraph_count}"] = clean_line
            paragraph_count += 1

    return {
        "metadata": metadata,
        "title": title,
        "content": content
    }

def main(config, is_eng=False):
    with open(config['json_links_file'], 'r', encoding='utf-8') as f:
        data = json.load(f)
        urls = data['links']

    for url in urls:
        html_content = fetch_url_content(url)
        headline, press_release, press_release_text_plain = extract_specific_content_v1(html_content)

        if not headline or not press_release:
            headline, press_release, press_release_text_plain = extract_specific_content_v2(html_content)

        if is_eng:
            date, time = extract_metadata_eng(press_release_text_plain)
        else:
            date, time = extract_metadata_chi(press_release_text_plain)

        markdown_content = convert_to_markdown(headline, press_release)

        filename = get_filename_from_url(url)
        markdown_path = os.path.join(config['markdown_output_folder'], filename)
        save_as_markdown(markdown_content, markdown_path)

        json_data = convert_markdown_to_json(markdown_content, date, time, url)
        json_filename = filename.replace('.md', '.json')
        json_path = os.path.join(config['json_output_folder'], json_filename)

        with open(json_path, 'w', encoding='utf-8') as json_file:
            json.dump(json_data, json_file, ensure_ascii=False, indent=4)

        print(f"✅ Saved Markdown: {markdown_path}")
        print(f"✅ Saved JSON: {json_path}")

# === Extract Panel Paper ===
def download_pdf(pdf_url, local_path):
    response = requests.get(pdf_url)
    response.raise_for_status()
    with open(local_path, 'wb') as f:
        f.write(response.content)
    print(f"✅ PDF downloaded to {local_path}")

def extract_text_from_pdf(local_path, converter):
    rendered = converter(local_path)
    text, _, _ = text_from_rendered(rendered)
    return text

def extract_tables_from_pdf(local_path, page_num):
    with pdfplumber.open(local_path) as pdf:
        if page_num < len(pdf.pages):
            page = pdf.pages[page_num]
            tables = page.extract_tables()
            table_areas = [table.bbox for table in page.find_tables()]
            return tables, table_areas
        return [], []

def split_into_paragraphs(text):
    paragraphs = [para for para in text.split("\n\n") if para.strip()]
    return {f"p{i+1}": para for i, para in enumerate(paragraphs)}

def save_to_json(metadata, title, content, json_path):
    data = {
        "metadata": metadata,
        "title": title,
        "content": content
    }
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def format_date_time(date_str):
    date_obj = datetime.strptime(date_str[2:15], "%Y%m%d%H%M%S")
    formatted_date = date_obj.strftime("%Y-%m-%d")
    formatted_time = date_obj.strftime("HKT %H:%M")
    return formatted_date, formatted_time

def extract_creation_date(local_path):
    with pdfplumber.open(local_path) as pdf:
        metadata = pdf.metadata
        return metadata.get("CreationDate", "")

def main_pdf(config):
    config_parser = ConfigParser(config)
    converter = PdfConverter(
        config=config_parser.generate_config_dict(),
        artifact_dict=create_model_dict(),
        processor_list=config_parser.get_processors(),
        renderer=config_parser.get_renderer(),
        llm_service=config_parser.get_llm_service()
    )

    with open(config['json_links_file'], 'r', encoding='utf-8') as f:
        data = json.load(f)
        pdf_urls = data.get("links", [])

    for pdf_url in pdf_urls:
        local_path = get_filename_from_url(pdf_url)
        download_pdf(pdf_url, local_path)

        try:
            text = extract_text_from_pdf(local_path, converter)
            paragraphs = split_into_paragraphs(text)
            content = paragraphs

            table_content = []
            for page_num in range(len(paragraphs)):
                tables, _ = extract_tables_from_pdf(local_path, page_num)
                if tables:
                    for table in tables:
                        for row in table:
                            for cell in row:
                                if cell and cell.strip():
                                    table_content.append(cell.strip())

            metadata = {
                "date": "",
                "start_time": "",
                "end_time": "",
                "time": "",
                "is_ocr": config.get("force_ocr", False),
                "table": ", ".join([key for key, para in paragraphs.items() if any(cell in para for cell in table_content)]),
                "source_link": pdf_url
            }

            creation_date = extract_creation_date(local_path)
            if creation_date:
                formatted_date, formatted_time = format_date_time(creation_date)
                metadata["date"] = formatted_date
                metadata["time"] = formatted_time

            title = ""
            original_filename = get_filename_from_url(pdf_url)
            json_filename = os.path.join(config['json_output_folder'], os.path.splitext(original_filename)[0] + '.json')
            save_to_json(metadata, title, content, json_filename)

            print(f"✅ Extracted content saved to {json_filename}")

        finally:
            if os.path.exists(local_path):
                os.remove(local_path)
                print(f"🗑️ PDF file {local_path} has been removed.")
# --- JSON Metadata Extraction Section ---
def load_source_mapping(config):
    df = pd.read_csv(config['source_code_file'])
    mapping = {}
    for _, row in df.iterrows():
        dataset_code_key = str(row['dataset_code']).strip()
        mapping[dataset_code_key] = {
            'source_name': str(row['source_name']).strip(),
            'source_code': str(row['source_code']).strip()
        }
    return mapping

def format_date(meta_date):
    if meta_date != 'N/A':
        try:
            parsed_date = datetime.strptime(meta_date, '%Y-%m-%d')
            return parsed_date.strftime('%Y%m%d')
        except ValueError:
            return 'N/A'
    return 'N/A'

def extract_json_metadata(json_path, year, dataset_code, source_info_mapping, config):
    with open(json_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    file_name = os.path.basename(json_path)
    file_size = os.path.getsize(json_path)
    full_path = os.path.abspath(json_path)

    root_path = config['root_path']
    data_file_path = full_path[len(root_path):] if full_path.startswith(root_path) else full_path
    if not data_file_path.startswith(r"\\"):
        data_file_path = "\\" + data_file_path

    file_path = os.path.join(root_path, data_file_path.lstrip("\\"))

    meta_date = data.get('metadata', {}).get('date', '')
    date = format_date(meta_date)

    source_info = source_info_mapping.get(dataset_code, {'source_name': 'N/A', 'source_code': 'N/A'})

    last_modified_timestamp = os.path.getmtime(json_path)
    last_modified_date = datetime.fromtimestamp(last_modified_timestamp).date()
    target_date = config['target_import_date']
    import_date = last_modified_date.strftime('%Y-%m-%d') if last_modified_date == target_date else 'N/A'

    return {
        "id": " ",
        'year': year,
        'date': date,
        'file_name': file_name,
        'file_size': file_size,
        'content_word_count': len(json.dumps(data).split()),
        "source_name": source_info['source_name'],
        'source_code': source_info['source_code'],
        'dataset_code': dataset_code,
        'original_file_format': "JSON",
        'use_file_format': "json",
        'ocr': "N",
        'root_path': root_path,
        'data_file_path': data_file_path,
        'file_path': file_path,
        'import_date': import_date,
        'last_modified': last_modified_timestamp
    }

def process_folder(config, source_info_mapping):
    metadata_list = []
    folder_path = config['folder_path']
    dataset_code = next((part for part in folder_path.split(os.sep) if part.startswith("hhb_")), "N/A")

    for root, _, files in os.walk(folder_path):
        year = os.path.basename(root)
        for file in files:
            if file.lower().endswith('.json'):
                file_path = os.path.join(root, file)
                metadata = extract_json_metadata(file_path, year, dataset_code, source_info_mapping, config)
                metadata_list.append(metadata)

    metadata_list.sort(key=lambda x: x['last_modified'])
    return metadata_list, dataset_code

def save_metadata_to_csv(metadata_list, csv_path):
    df = pd.DataFrame(metadata_list)
    df.drop(columns=['last_modified'], inplace=True)
    df['import_date'] = df['import_date'].astype(str)
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')

def generate_csv_filename(folder_path, dataset_code):
    folder_parts = folder_path.split(os.sep)
    try:
        dataset_index = folder_parts.index(dataset_code)
        after_parts = folder_parts[dataset_index + 1:]
        chi_folder = "chi" if "chi" in [p.lower() for p in after_parts] else ""
        after_dataset_folder = next((p for p in after_parts if p.lower() != "chi"), "unknown")
    except ValueError:
        chi_folder = ""
        after_dataset_folder = "unknown"

    prefix = f"{chi_folder}_" if chi_folder else ""
    return f"{prefix}{after_dataset_folder}_{dataset_code}_metadata.csv"

# --- Excel Masterlist Update Section ---
def update_excel_masterlist(excel_config, target_date):
    excel_path = excel_config["excel_file"]
    sheet_configs = excel_config["sheets"]

    wb = load_workbook(excel_path)
    excel_data = pd.read_excel(excel_path, sheet_name=None, engine="openpyxl")

    for sheet_config in sheet_configs:
        sheet_name = sheet_config["sheet_name"]
        metadata_folder = sheet_config["metadata_folder"]
        compare_columns = sheet_config["compare_columns"]

        pattern = os.path.join(metadata_folder, "*.csv")
        files = glob.glob(pattern)
        dfs = [pd.read_csv(file).dropna(how="all") for file in files]
        metadata_df = pd.concat(dfs, ignore_index=True)

        metadata_df.rename(columns={"dataset_code": "datset_code"}, inplace=True)
        metadata_df["import_date"] = pd.to_datetime(metadata_df["import_date"], errors="coerce")
        metadata_df = metadata_df[metadata_df["import_date"] == pd.to_datetime(target_date)]
        metadata_df = metadata_df.dropna(subset=["file_name", "datset_code"])
        metadata_df = metadata_df[
            (metadata_df["file_name"].str.strip() != "") &
            (metadata_df["datset_code"].str.strip() != "")
        ]

        if metadata_df.empty:
            print(f"❌ No valid data for sheet {sheet_name} on {target_date}")
            continue

        master_df = excel_data[sheet_name]

        for col in compare_columns:
            if col in metadata_df.columns:
                metadata_df[col] = metadata_df[col].astype(str).str.strip().str.lower()
            if col in master_df.columns:
                master_df[col] = master_df[col].astype(str).str.strip().str.lower()

        merged_df = metadata_df.merge(master_df[compare_columns], on=compare_columns, how="left", indicator=True)
        new_rows_df = merged_df[merged_df["_merge"] == "left_only"].drop(columns=["_merge"])

        if new_rows_df.empty:
            print(f"✅ No new data added to sheet {sheet_name}")
            continue

        if "id" in master_df.columns:
            master_df["id"] = pd.to_numeric(master_df["id"], errors="coerce")
            max_id = master_df["id"].dropna().astype(int).max()
            new_ids = range(max_id + 1, max_id + 1 + len(new_rows_df))
        else:
            new_ids = range(1, 1 + len(new_rows_df))
        new_rows_df["id"] = new_ids

        for row in new_rows_df.itertuples(index=False):
            wb[sheet_name].append(row)

        print(f"✅ Appended {len(new_rows_df)} new rows to sheet {sheet_name}")

    wb.save(excel_path)
    print("✅ All updates completed and saved to Excel file.")

# --- CSV Masterlist Update Section ---
class MetadataProcessor:
    def __init__(self, config):
        self.config = config
        self.metadata_df = pd.DataFrame()
        self.master_df = pd.read_csv(config['master_file'])

    def load_metadata(self):
        pattern = os.path.join(self.config['metadata_folder'], "*.csv")
        files = glob.glob(pattern)
        dfs = [pd.read_csv(file).dropna(how="all") for file in files]
        self.metadata_df = pd.concat(dfs, ignore_index=True)

    def clean_metadata(self):
        self.metadata_df.rename(columns={"dataset_code": "datset_code"}, inplace=True)
        if "import_date" in self.metadata_df.columns:
            self.metadata_df["import_date"] = pd.to_datetime(self.metadata_df["import_date"], errors="coerce")
            self.metadata_df = self.metadata_df[self.metadata_df["import_date"] == pd.to_datetime(self.config['target_date'])]
            self.metadata_df = self.metadata_df.dropna(subset=["file_name", "datset_code"])
            self.metadata_df = self.metadata_df[
                (self.metadata_df["file_name"].str.strip() != "") &
                (self.metadata_df["datset_code"].str.strip() != "")
            ]

    def normalize_columns(self, df, columns):
        for col in columns:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip().str.lower()
        return df

    def find_new_rows(self):
        metadata_df = self.normalize_columns(self.metadata_df, self.config['compare_columns'])
        master_df = self.normalize_columns(self.master_df, self.config['compare_columns'])
        merged_df = metadata_df.merge(master_df[self.config['compare_columns']], on=self.config['compare_columns'], how="left", indicator=True)
        new_rows_df = merged_df[merged_df["_merge"] == "left_only"].drop(columns=["_merge"])
        new_rows_df = new_rows_df.dropna(how="all")
        new_rows_df = new_rows_df.dropna(subset=["file_name", "datset_code"])
        new_rows_df = new_rows_df[
            (new_rows_df["file_name"].str.strip() != "") &
            (new_rows_df["datset_code"].str.strip() != "")
        ]
        return new_rows_df

    def assign_new_ids(self, new_rows_df):
        if "id" in self.master_df.columns:
            self.master_df["id"] = pd.to_numeric(self.master_df["id"], errors="coerce")
            max_id = self.master_df["id"].dropna().astype(int).max()
            new_ids = range(max_id + 1, max_id + 1 + len(new_rows_df))
        else:
            new_ids = range(1, 1 + len(new_rows_df))
        new_rows_df["id"] = new_ids
        return new_rows_df

    def save_updated_master(self, new_rows_df):
        updated_df = pd.concat([self.master_df, new_rows_df], ignore_index=True)
        if "import_date" in updated_df.columns:
            updated_df["import_date"] = pd.to_datetime(updated_df["import_date"], errors="coerce").dt.strftime("%Y-%m-%d")
        updated_df.to_csv(self.config['output_file'], index=False)
        return len(new_rows_df)

    def run(self):
        self.load_metadata()
        self.clean_metadata()

        if self.metadata_df.empty:
            print("❌ No valid data matching the import_date criteria")
            print("✅ All updates completed and saved to Excel file.")
            return

        new_rows_df = self.find_new_rows()
        if new_rows_df.empty:
            print("✅ No new data added")
            return

        new_rows_df = self.assign_new_ids(new_rows_df)
        print("✅ Newly added data as follows：")
        print(new_rows_df)
        new_rows_df.to_csv("newly_added_data.csv", index=False)
        added_count = self.save_updated_master(new_rows_df)
        print(f"✅ Appended {added_count} missing rows from metadata to master list.")

# --- Main Entry Point ---
def get_latest_friday(today):
    offset = (today.weekday() - 4) % 7
    return today - timedelta(days=offset)


def main_metadata():
    print("Start running create JSON Metadata Files")

    with open("json_metadata_config.json", 'r', encoding='utf-8') as f:
        json_config_data = json.load(f)

    json_config = {
        "output_folder": json_config_data.get("output_folder", "./output"),
        "root_path": json_config_data.get("root_path", "./data"),
        "source_code_file": json_config_data.get("source_code_file", "source_code.csv"),
        "target_import_date": datetime.strptime(json_config_data.get("target_import_date", "2025-10-17"), "%Y-%m-%d").date()
    }

    folder_paths = []
    for key, value in json_config_data.items():
        if isinstance(value, list):
            folder_paths.extend([v for v in value if isinstance(v, str) and v.startswith("\\\\")])

    os.makedirs(json_config['output_folder'], exist_ok=True)
    source_info_mapping = load_source_mapping(json_config)

    for folder_path in folder_paths:
        json_config['folder_path'] = folder_path
        metadata_list, dataset_code = process_folder(json_config, source_info_mapping)
        csv_filename = generate_csv_filename(folder_path, dataset_code)

        is_chi = csv_filename.lower().startswith("chi_")
        parts = csv_filename.replace("chi_", "").split("_")
        main_code = parts[0] if parts else "unknown"

        language_folder = "chi" if is_chi else "eng"
        target_folder_name = f"{main_code}_{language_folder}"

        subfolder_path = os.path.join(json_config['output_folder'], target_folder_name)
        os.makedirs(subfolder_path, exist_ok=True)

        csv_path = os.path.join(subfolder_path, csv_filename)
        save_metadata_to_csv(metadata_list, csv_path)
        print(f"✅ Metadata saved to: {csv_path}")

    with open("update_masterlist.json", "r", encoding="utf-8") as f:
        config = json.load(f)

    latest_friday = get_latest_friday(datetime.today()).strftime("%Y-%m-%d")

    print("Start running update Masterlist Excel")
    update_excel_masterlist(config["excel_config"], latest_friday)

    print("Start running update CSV Master list")
    for csv_config in config["csv_configs"]:
        csv_config["target_date"] = latest_friday
        csv_config["output_file"] = csv_config.get("output_file") or "updated_" + os.path.basename(csv_config["master_file"])
        processor = MetadataProcessor(csv_config)
        processor.run()


if __name__ == "__main__":
    with open("raw_data_config.json", "r", encoding="utf-8") as f:
        config_data = json.load(f)

    def run_task(section_name, lang_key):
        if section_name in config_data:
            section_config = config_data[section_name][0]
            config = {
                "json_links_file": section_config[f"{lang_key}_json_links_file"],
                "markdown_output_folder": section_config["markdown_output_folder"],
                "json_output_folder": section_config[f"{lang_key}_json_output_folder"]
            }
            print(f"\n🚀 Processing {lang_key.upper()} {section_name.replace('_', ' ')}...")
            os.makedirs(config['markdown_output_folder'], exist_ok=True)
            os.makedirs(config['json_output_folder'], exist_ok=True)
            main(config, is_eng=(lang_key == "eng"))

    # === Step 1: Extract HTML-based content ===
    run_task("press_release", "chi")
    run_task("press_release", "eng")
    run_task("speech", "chi")
    run_task("speech", "eng")
    run_task("legco_qna", "chi")
    run_task("legco_qna", "eng")

    # === Step 2: Extract PDF-based Panel Papers ===
    legco_config = config_data["legco_panel_paper"][0]

    chi_config = {
        "output_format": "markdown",
        "force_ocr": False,
        "llm_service": "marker.services.gemini.GoogleGeminiService",
        "json_links_file": legco_config["chi_json_links_file"],
        "json_output_folder": legco_config["chi_json_output_folder"]
    }
    print("\n🚀 Processing LEGCO Panel Paper - Chinese")
    os.makedirs(chi_config['json_output_folder'], exist_ok=True)
    main_pdf(chi_config)

    eng_config = {
        "output_format": "markdown",
        "force_ocr": False,
        "llm_service": "marker.services.gemini.GoogleGeminiService",
        "json_links_file": legco_config["eng_json_links_file"],
        "json_output_folder": legco_config["eng_json_output_folder"]
    }
    print("\n🚀 Processing LEGCO Panel Paper - English")
    os.makedirs(eng_config['json_output_folder'], exist_ok=True)
    main_pdf(eng_config)

    # === Step 3: Extract JSON Metadata & Update Masterlists ===
    print("\n🚀 Running Metadata Extraction and Masterlist Update")
    main_metadata()
