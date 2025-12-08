# Data Collection Module of LLM CCRM Data Pipeline

## Overview
This module scrapes publicly available documents from a list of URLs, extracts metadata, converts PDFs via OCR if necessary, splits the content into paragraphs, and stores the processed data in a JSON file with a Markdown‑style format. The module can be deployed in Docker and served via flask API.

## API Endpoints
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/run` | POST | Initiates the full pipeline. Expects JSON body with a `callback_url` where a status notification will be POSTed once the job completes.
| `/process_data` | POST | Triggers only the post‑processing step (markdown→JSON conversion). Same `callback_url` mechanism.

### Request body
```json
{
  "callback_url": "https://example.com/webhook"
}
```

## Configuration
JSON configuration files located in the working directory (or mounted volume):
- `panel_paper_link_config.json`
- `filtering_link_config.json`
- `raw_data_config.json`
- `update_masterlist.json`
- `json_metadata_config.json`
