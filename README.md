# Drug Research Assistant

An AI-powered assistant for analyzing drug research papers using Streamlit and LangChain. This tool helps researchers and medical professionals quickly analyze and understand drug-related research papers.

## Features

- **AI-Powered Analysis**: Utilizes TinyLlama and LangChain for intelligent paper analysis
- **Chemical Structure Analysis**: Molecular property calculation and visualization using RDKit
- **Multi-Source Data**: Integrates with PubMed, DrugBank, and ClinicalTrials.gov
- **Advanced Analytics**: Statistical analysis of clinical trials and drug effectiveness
- **Interactive UI**: User-friendly interface with data visualization and export options

## Installation

1. Clone the repository:
```bash
git clone https://github.com/YourUsername/drug-research-assistant.git
cd drug-research-assistant
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys
```

2. Run the Streamlit app:
```bash
streamlit run src/drug_research_assistant.py
```

## Requirements

- Python 3.8+
- M1 Mac compatible
- 4GB+ RAM recommended

## Development

1. Install development dependencies:
```bash
pip install -r requirements-dev.txt
```

2. Run tests:
```bash
pytest tests/
```

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

MIT License - see LICENSE file for details

## Acknowledgments

- PubMed for research data access
- Streamlit for the UI framework
- LangChain for AI capabilities