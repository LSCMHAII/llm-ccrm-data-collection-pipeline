# Data Collection and Extraction Module of LLM CCRM Data Pipeline

**LLM CCRM Data Pipeline**

The Data Collection and Extraction Module automates scraping, parsing, transformation, and organization of publicly available information from official Hong Kong Government Websites **(Health Bureau)** and the Legislative Council **(LegCo)**. 

As a core component of the CCRM project, this module ensures that downstream LLM processing and system integration receive clean, structured Markdown and JSON data with complete metadata.

## 📂 Table of Contents
- [Purpose](#purpose)
- [High-Level Architecture](#high-level-architecture)
- [Code Repository](#code-repository)
- [Configuration Files](#configuration-files)
- [Installation and Setup](#installation-and-setup)
- [System Integration & API Specs](#system-integration--api-specs)
- [Workflow and Script](#workflow-and-script)
- [Output Types](#output-types)
- [Reusable Modules and Customization](#reusable-modules-and-customization)
- [Current Limitations](#current-limitations)


## Purpose
This pipeline was built to automatically collect, clean, and structure official HK government and LegCo data so it can be directly fed into LLM and CCRM systems.

#### Target sources:
- `HTML` - Press Release, Speech, LegCo Q&A
- `PDF` - Meeting papers (LegCo Panel on Health Services)

#### Core Output: 
- Clean Markdown + structured JSON + metadata CSVs + Excel masterlist.

## High-Level Architecture
The following diagram illustrates the comprehensive data flow, from multi-source scraping to the final JSON/Markdown output categorization:
<img width="3864" height="1141" alt="RawDataPipeLine drawio (1)" src="https://github.com/user-attachments/assets/579a24d6-16a9-471d-b5bc-a0e2e54145a9" />

#### Key Processing Logic (as shown in diagram):
- Multi-Source Ingestion: Handles diverse sources including LegCo Panel Papers, SFC, Health Bureau Press Releases, and VHIS documents.
- Format-Specific Extraction:
  - **HTML:** Parsed via beautifulsoup4 for clean text and table extraction.
  - **PDF:** Processed via pdf-marker to ensure high-fidelity reading order and table mapping.

## Code Repository
All code is in `llm-ccrm-data-collection-pipeline` repository on GitHub.

**Clone via HTTPS:**
```bash
 https://github.com/LSCMHAII/llm-ccrm-data-collection-pipeline.git
```

**Clone via GitHub CLI:**
```bash
gh repo clone LSCMHAII/llm-ccrm-data-collection-pipeline
```

| Core Script | Primary Responsibility  | 
|----------|----------|
| `main.py`   | URL collection, file downloading, and initial JSON staging. |
| `process_data.py` | HTML/PDF parsing, Markdown conversion, and metadata updates. |

Dependencies are listed in `requirements.txt`

## Configuration Files
All configuration must reside in the working directory or be mounted via Docker volumes.

| Config File | Responsibility  |
|----------|----------|
| `link_config.json` | Source URLs for Press Releases, Speeches, LegCo Q&A (Chinese + English) |
| `panel_paper_link_config.json` |  URLs for LegCo Panel on Health Services papers  |
| `filtering_link_config.json` |  Link history for deduplication.  |
| `raw_data_config.json` |  Extraction rules, output directories, PDF parameters  |
| `update_masterlist.json` |  Excel masterlist update rules (sheets, columns) |
| `json_metadata_config.json ` |  Metadata CSV generation settings  |


## Installation and Setup
### Prerequisites

| Dependency | Requirements     |  Notes   |
|------------|------------------|----------|
| Python     |  ≥ 3.11          | Tested with 3.11 |
| pip        |  Lastest version |  `pip install --upgrade pip` |
| Python packages | See `requirements.txt` |      -            |
| Working Directory | Config files + data folder |  Must contain all `.json `config files |
| Optional: Docker |  Docker + Docker Compose | For containerized deployment  |


### Step-by-Step Installation
**1. Navigate to the project folder**
```bash
cd /path/to/your/project
```
**2. Create and activate a virtual environment** (strongly recommended to avoid conflicts)
```bash
python -m venv venv
# Linux / macOS
source venv/bin/activate
# Windows
venv\Scripts\activate
```
**3. Install all Python dependencies**
```bash
# Install the packages required
pip install -r requirements.txt
```
**4. Verify Installation:**
```bash
python -c "import bs4, selenium, marker, pdfplumber, pandas; print('✅ Core environment ready!')"
```
**5. (Optional) Docker users**
- Build the image: `docker build -t llmccrm-data-collection `.
- Run with mounted volume:
```bash
docker run -v $(pwd)/config:/app/config \
           -v $(pwd)/data:/app/data \
           llmccrm-data-collection
```

**6. (Optional) Run as Flask API**

The module can be started as a web service (see your `server.py` or Flask setup in the project).

### Major Packages & Their Purpose
| Packages | Responsibility  |
|----------|----------|
| beautifulsoup4 |  HTML scraping and parsing |
| selenium + webdriver-manager | Dynamic page handling (PDF link extraction) |
| marker-pdf  | Advanced PDF parsing with LLM support |
| pdfplumber  | Table detection in PDFs |
| markdownify | HTML → Markdown conversion  |
| pandas + openpyxl | Excel masterlist updates  |
| requests + lxml | HTTP requests and XML/HTML processing |
| flask | Optional API service  |

## System Integration & API Specs
This module is designed to operate within the project's Docker container and provides an API interface for n8n automation.

**1. n8n API Interface**

To trigger the module from n8n (HTTP Request Node), use the following endpoints:
| Endpoint | Method | Description |
|----------|----------|-----------|
| /run  | POST | Initiates the full pipeline (Scraping + Processing).|
| /process_data | POST | Triggers only the post-processing step (Markdown to JSON).|

**Request Body:**
```JSON
{
  "callback_url": "https://your-n8n-webhook-url.com"
}
```

**2. Container Environment Specs**

The module expects the following environment configuration for seamless integration:
- **Port Mapping**: The Flask API is exposed on port `5000` by default.
- **Volume Mounting**: Requires mounting `/config` for JSON settings and `/data` for extraction outputs.
- **Environment Variables**: Ensure `GEMINI_API_KEY` is provided for PDF processing.

## Workflow and Script
**Stage 1: Extraction (main.py)**

Orchestrates three-stage workflow:
- Scrape Press Releases / Speeches / Q&A → JSON
- Extract PDF links via Selenium
- Filter, deduplicate, update master lists

**Run:**
```bash
python main.py
```

**Stage 2: Processing (process_data.py)**

Handles content transformation, OCR/Table detection, and final data export.
1. HTML → Markdown + cleaned JSON
2. PDF download & parsing (LLM/OCR + table detection)
3. Metadata CSV generation + Excel masterlist update
```bash
python process_data.py
```
**Example with parameters:**
```python
main_combined(base_path="/app/nas_data/pipeline_data_collection", target_date="2025-09-12")
```
## Output Types
| Type  | Format  | Content |
|-------|---------|---------|
| HTML extraction | `.md` + `.json` | Markdown files + structured JSON (title, content, metadata) |
| PDF extraction  | `.json `  | Paragraphs + table mappings + metadata  |
| Metadata  | `.csv`  | Dataset-level metadata (year, word count, paths, import_date) |
| Masterlist  | `.xlsx `(multiple sheets)  | Append-only updates based on compare_columns |

## Reusable Modules and Customization
The following components are modular by design and can be directly imported, reused, or extended in other projects without running the full pipeline.

### 1. Functional Categorization of Reusable Modules
The following functions in process_data.py are built for standalone utility in other data engineering or LLM projects:

| Category | Module / Function | Description | Direct Reuse Conditions  |
|-----------|-------------------|-------------|-------------------------|
| File I/O | `download_pdf()` |   Download a PDF with a proper `User-Agent`, configurable timeout, and ensure output directory exists. | Any task that needs a local PDF copy before parsing |
| PDF Extraction | `extract_text_from_pdf()`  | Use **Marker** to render and extract near-reading-order text from PDFs  | LLM-ready text for downstream chunking or indexing  |
| PDF Extraction  | `extract_tables_from_pdf()` | Uses pdfplumber to detect and extract tables and preserves row/column structure for structured output  |  The PDF contains tabular data |
| Data Processing | `split_into_paragraphs()` | Normalize and segment raw text into paragraph-level JSON blocks  | Clean text (HTML/PDF/LLM output) |
| Metadata Mining| `extract_creation_date ()`  | Extract PDF doucument metadata  | PDF has usable metadata |
| Data Storage  | `save_to_json()`  | Saves structured extraction results to JSON | Structured Python dicts |


### 2. Customization Guide
These modules are built to be easily extended by modifying specific logic or JSON configurations:

**1. HTML Scraping and Parsing Rules**
- **Targeting New Sites:** 
    - The primary extractor currently targets `#PRHeadlineSpan`, `#pressrelease` and `#content`.
- **How to customize:**
    - You can add new DOM selectors or create an "Extractor v3" in process_data.py to support different government or official websites.

**2. Structured Content Handling**
- **Currently State:**
    - Lists and tables are flattened into paragraph blocks by default.
- **How to customize:**
    - Modify `convert_to_markdown()` to preserve `<ul>`, `<ol>`, or `<table>` as Markdown lists or special keys (e.g. `list_1`, `table_1`)

**3.Metadata & Date Extraction**
- **Current State:**
    - Supports specific Chinese formats and English formate
      ```bash
      2025年3月12日（星期三）
      香港時間14時30分
      ```
      ```bash
      Ends/Issued at
      Issued at HKT
      ```
- **How to customize:**
    -  Add new regex patterns in `extract_metadata_chi()` or `extract_metadata_eng()` to support additional timezones or document styles.

**4. PDF Extraction**
- Uses Marker with Gemini service (`force_ocr` option)
- Tables detected via pdfplumber
- Customize: Switch LLM service, add hybrid OCR logic, or change table mapping strategy

**5. Output & Masterlist Management**
- **Schema:**
  - The JSON output can be customized to include fields like language or processing_version.
- **Default Structure:**
 ```json
{
  "metadata": {
    "date": "2025-03-12",
    "time": "HKT 14:00",
    "is_ocr": false,
    "source_link": "https://..."
  },
  "title": "Press Release Title",
  "content": { 
    "p1": "...", 
    "p2": "..." 
    }
}
```
- **Excel Logic:**
  -  The masterlist is currently append-only.
- **How to customize:**
  - You can update `update_masterlist.json` to define new columns or versioning rules.

**6. General Improvements**
- Temporary PDFs are deleted immediately
- Customize: Add dedicated temp folder, checksum validation, or persist original PDFs for audit

All customization can be done by editing the respective functions in `process_data.py` or updating the JSON config files.


## Current Limitations
| Limitation  | Description |
|-------------|-------------|
| Website fragility | Fixed DOM selectors (e.g. #PRHeadlineSpan) – breaks if government sites change  |
| Performance | Selenium for dynamic pages → slow, high CPU/memory usage |
| External dependency | PDF parsing relies on Google Gemini API (cost + rate limits) or OCR fallback |
| Storage | Temporary PDFs are deleted immediately (no built-in audit trail) |
| Configuration | Adding new sources requires editing multiple JSON config files  |
| Masterlist  | Append-only Excel updates (no delete or versioning) |
| Scalability | No parallel processing or real-time scheduling  |

