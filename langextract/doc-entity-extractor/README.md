# 🔍 LangExtract Document Entity Extractor

A Streamlit application that uses Google's LangExtract to transform unstructured text into highlighted, structured information with entity extraction and visualization.

## � About LangExtract

This application is powered by [Google's LangExtract](https://github.com/google/langextract), an open-source library for extracting structured information from unstructured text using large language models.

**LangExtract Official Repository**: https://github.com/google/langextract

## �🚀 Quick Start

### Prerequisites

- Python 3.8+
- [uv](https://docs.astral.sh/uv/) package manager
- Google Gemini API key

### 1. Clone the Repository

```bash
git clone https://github.com/piyushagni5/langgraph-ai.git
cd langgraph-ai/langextract/doc-entity-extractor
```

### 2. Install Dependencies

Using `uv` (recommended):

```bash
uv pip install -r requirements.txt
```

Or using standard pip:

```bash
pip install -r requirements.txt
```

### 3. Set Up API Key

Create a `.env` file in the project root:

```bash
echo "GOOGLE_API_KEY=your_gemini_api_key_here" > .env
```

Or set up Streamlit secrets by creating `.streamlit/secrets.toml`:

```toml
GOOGLE_API_KEY = "your_gemini_api_key_here"
```

### 4. Run the Application

```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

## 📚 Features

- **Entity Extraction**: Automatically extract entities from sample documents
- **Interactive Highlighting**: View extracted entities highlighted in original text
- **Query-Based Filtering**: Search for specific types of information
- **Export Capabilities**: Download visualization as HTML
- **Real-time Processing**: Process multiple documents simultaneously

## 🔧 Usage

1. **Start the Application**: Run `streamlit run app.py`
2. **Provide API Key**: Enter your Google Gemini API key when prompted
3. **Optional Query**: Enter a specific query to filter results (e.g., "company founders", "financial information")
4. **Process Documents**: Click "🚀 Process Documents & Extract Information"
5. **Explore Results**: Browse through the three tabs:
   - **Highlighted Text**: View entities highlighted in source text
   - **Entity Summary**: See structured extraction results
   - **Search Results**: View query-specific filtered entities

## 💡 Suggested Queries to Try

Once the Streamlit UI is up and running, try these example queries in the query input field, then click "🚀 Process Documents & Extract Information":

### 🏢 Company Information
- `company founders and locations`
- `tech companies headquarters`
- `companies founded in 1970s`

### 👥 People & Leadership
- `founders and CEOs`
- `Steve Jobs Apple`
- `current leadership teams`

### 💰 Financial Information
- `revenue and financial data`
- `company valuations`
- `financial performance`

### 📅 Historical Data
- `founding dates and years`
- `company history timeline`
- `when companies went public`

### 🌍 Geographic Information
- `California companies`
- `Silicon Valley locations`
- `company headquarters locations`

### 🔍 General Extraction
- Leave the query field **empty** to extract all entities from the sample documents

> **Tip**: The sample documents contain information about Apple, Microsoft, and Google, so queries related to these companies will yield the most relevant results!

## 📁 Project Structure

```
doc-entity-extractor/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Project dependencies
├── .env                  # Environment variables (create this)
├── src/
│   ├── entity_extractor.py    # LangExtract processing logic
│   ├── display_manager.py     # Streamlit UI components
│   └── utils.py               # Utility functions
├── data/
│   └── sample_documents.py    # Sample text documents
├── templates/
│   └── few_shot_examples.py   # Example templates
└── .streamlit/
    └── secrets.toml           # Streamlit secrets (alternative to .env)
```

## 🔑 Getting a Gemini API Key

1. Visit [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Sign in with your Google account
3. Click "Create API Key"
4. Copy the generated key and add it to your `.env` file

## 📝 License

This project is part of the langgraph-ai repository. See the main repository for license information.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues or pull requests to the main repository.