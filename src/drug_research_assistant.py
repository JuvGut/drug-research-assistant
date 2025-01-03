# src/drug_research_assistant.py

import streamlit as st
import pandas as pd
import asyncio
import base64
from datetime import datetime
from typing import List, Dict, Optional
import io
import json
import torch
from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from bs4 import BeautifulSoup
import psutil
from dotenv import load_dotenv
import uvloop
import time

load_dotenv()

from config import Config
from models import MoleculeAnalyzer
from utils import Cache, APIClient

def get_device():
    """Determine the best available device"""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        # Check MPS backend status
        if torch.backends.mps.is_built():
            try:
                # Test MPS availability
                torch.zeros(1).to("mps")
                return "mps"
            except Exception:
                st.warning("MPS (Metal Performance Shaders) is available but not working properly. Falling back to CPU.")
                return "cpu"
    return "cpu"

def check_available_memory():
    """Check if enough memory is available for model loading"""
    memory = psutil.virtual_memory()
    available_gb = memory.available / (1024 * 1024 * 1024)  # Convert to GB
    return available_gb

@st.cache_resource(ttl=3600)
def load_llm():
    """Initialize the language model with proper accelerate integration"""
    try:
        # Check available memory before loading
        available_memory = check_available_memory()
        if available_memory < 4:  # Require at least 4GB free
            st.error(f"Insufficient memory available ({available_memory:.1f}GB). Need at least 4GB.")
            return None
        
        model_id = "facebook/opt-125m" #"TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        device = get_device()
        st.info(f"Using device: {device}")
        
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        
        # Load model with accelerate
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            #torch_dtype=torch.float32,  # Using float32 for better compatibility
            torch_dtype=torch.float16,  # Using float16 for faster inference
            device_map="auto"
        )
        
        # Create pipeline without specifying device
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_length=2048, 
            temperature=0.3,
            top_p=0.95,
            repetition_penalty=1.15
        )
        return HuggingFacePipeline(pipeline=pipe)
    except Exception as e:
        st.error(f"Error loading language model: {str(e)}")
        return None

@st.cache_resource
def load_embeddings():
    """Initialize the embeddings model with proper error handling"""
    try:
        device = get_device()
        model_kwargs = {'device': device if device != "mps" else "cpu"}
        
        return HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs=model_kwargs,
            encode_kwargs={'device': device if device != "mps" else "cpu"}
        )
    except Exception as e:
        st.error(f"Error loading embeddings model: {str(e)}")
        return None

class DrugResearchAssistant:
    def __init__(self):
        self.config = Config()
        self.cache = Cache(self.config.CACHE_DIR, self.config.CACHE_EXPIRY)
        self.api_client = APIClient(self.config.PUBMED_BASE_URL, self.config.PUBMED_API_KEY)
        self.molecule_analyzer = MoleculeAnalyzer()
        self.initialize_models()

    def initialize_models(self):
        """Initialize AI models with proper error handling"""
        try:
            with st.spinner("Loading AI models..."):
                self.llm = load_llm()
                self.embeddings = load_embeddings()
                self.initialized = bool(self.llm and self.embeddings)
                
                if self.initialized:
                    st.success("Models loaded successfully!")
                else:
                    st.error("Failed to initialize models")
        except Exception as e:
            st.error(f"Error initializing models: {str(e)}")
            self.initialized = False

    async def fetch_pubmed_data(self, drug_name: str) -> List[Dict]:
        """Fetch research papers from PubMed"""
        try:
            # First, search for papers (JSON endpoint)
            search_params = {
                'db': 'pubmed',
                'term': drug_name,
                'retmax': str(self.config.MAX_RESULTS),
                'format': 'json'
            }
        
            search_response = await self.api_client.get(
                'esearch.fcgi',
                search_params,
                response_format='json'
            )
        
            if not search_response.get('esearchresult', {}).get('idlist', []):
                return []
            
            # Fetch paper details (XML endpoint)
            ids = search_response['esearchresult']['idlist']
            fetch_params = {
                'db': 'pubmed',
                'id': ','.join(ids),
                'rettype': 'abstract',
                'retmode': 'xml'
            }
            
            xml_response = await self.api_client.get(
                'efetch.fcgi',
                fetch_params,
                response_format='xml'
            )
            
            # Parse XML response
            soup = BeautifulSoup(xml_response, 'xml')
            results = []
        
            for article in soup.find_all('PubmedArticle'):
                try:
                    title_elem = article.find('ArticleTitle')
                    title = title_elem.text if title_elem else ''
                    
                    abstract_elem = article.find('Abstract')
                    abstract = abstract_elem.text if abstract_elem else ''
                    
                    if title and abstract:
                        results.append({
                            'title': title,
                            'text': abstract
                        })
                except Exception as e:
                    st.warning(f"Error parsing article: {str(e)}")
                    continue
            
            return results
            
        except Exception as e:
            st.error(f"Error fetching PubMed data: {str(e)}")
            return []

    async def fetch_drug_data(self, drug_name: str) -> Dict:
        """Fetch comprehensive drug data from multiple sources"""
        try:
            # Check cache first
            cache_key = f"drug_data_{drug_name}"
            cached_data = self.cache.get(cache_key)
            if cached_data:
                return cached_data

            # Create a placeholder for the status
            status_placeholder = st.empty()
            
            # Fetch data from multiple sources concurrently
            status_placeholder.text("Fetching PubMed data...")
            pubmed_task = self.fetch_pubmed_data(drug_name)
            
            status_placeholder.text("Fetching clinical trials data...")
            trials_task = self.fetch_clinical_trials(drug_name)
            
            # Wait for all tasks to complete
            results = await asyncio.gather(
                pubmed_task, 
                trials_task,
                return_exceptions=True
            )
            
            # Process results
            status_placeholder.text("Processing results...")
            drug_data = {
                'pubmed_data': results[0] if not isinstance(results[0], Exception) else [],
                'clinical_trials': results[1] if not isinstance(results[1], Exception) else [],
            }
            
            # Generate overview using the new method
            if drug_data['pubmed_data']:
                drug_data['overview'] = self.generate_drug_overview(drug_data['pubmed_data'])
            
            # Cache the results
            self.cache.set(cache_key, drug_data)
            
            # Clear the status
            status_placeholder.empty()
            
            return drug_data
            
        except Exception as e:
            st.error(f"Error fetching drug data: {str(e)}")
            return {}

    def analyze_molecular_properties(self, smiles: str) -> Dict:
        """Analyze molecular properties of the drug"""
        try:
            if not self.molecule_analyzer.load_smiles(smiles):
                return {'error': 'Invalid SMILES notation'}
                
            properties = self.molecule_analyzer.calculate_properties()
            lipinski = self.molecule_analyzer.check_lipinski()
            
            # Get image as bytes
            image_bytes = self.molecule_analyzer.generate_2d_image()
            
            return {
                'properties': properties,
                'lipinski': lipinski,
                'structure_image': image_bytes
            }
        except Exception as e:
            return {'error': str(e)}

    def generate_drug_overview(self, pubmed_data: List[Dict]) -> str:
        """Generate a comprehensive drug overview from PubMed data"""
        if not pubmed_data:
            return "No overview data available"
            
        # Combine all abstracts
        all_text = " ".join([paper['text'] for paper in pubmed_data])
        
        prompt = f"""Based on the following research data, provide a concise overview that includes:
        1. The drug's primary therapeutic uses
        2. Its mechanism of action
        3. Key benefits and potential risks
        
        Research data: {all_text[:1000]}...
        """
        
        try:
            if self.llm:
                response = self.llm(prompt)
                return response
            return "Language model not available for overview generation"
        except Exception as e:
            return f"Error generating overview: {str(e)}"

    def explain_lipinski_rules(self) -> str:
        """Provide explanation of Lipinski's Rule of Five"""
        return """
        Lipinski's Rule of Five helps predict how likely a drug is to be absorbed by the body. The rules are:
        • Molecular weight ≤ 500 daltons
        • Lipophilicity (LogP) ≤ 5
        • Hydrogen bond donors ≤ 5
        • Hydrogen bond acceptors ≤ 10
        
        Meeting these criteria suggests better drug-like properties and oral bioavailability.
        """

    def summarize_research_findings(self, papers: List[Dict]) -> List[Dict]:
        """Generate a summary of research findings"""
        if not papers:
            return "No research findings available"
            
        summary = []
        for paper in papers[:3]:  # Summarize top 3 papers
            prompt = f"""Provide a 2-3 sentence summary of the key findings from this research:
            Title: {paper['title']}
            Abstract: {paper['text']}
            """
            
            try:
                if self.llm:
                    response = self.llm(prompt)
                    summary.append({
                        'title': paper['title'],
                        'summary': response
                    })
            except Exception as e:
                st.warning(f"Error summarizing paper: {str(e)}")
                
        return summary

    async def fetch_clinical_trials(self, drug_name: str) -> List[Dict]:
        """Enhanced clinical trials data fetching"""
        try:
            # For demo purposes, return mock data
            return [
                {
                    'trial_id': 'NCT12345678',
                    'phase': 'Phase 3',
                    'status': 'Active',
                    'participants': 500,
                    'start_date': '2023-01-15',
                    'completion_date': '2024-12-31',
                    'description': 'Randomized controlled trial evaluating efficacy...'
                },
                {
                    'trial_id': 'NCT87654321',
                    'phase': 'Phase 2',
                    'status': 'Recruiting',
                    'participants': 200,
                    'start_date': '2024-03-01',
                    'completion_date': '2025-06-30',
                    'description': 'Multi-center study investigating safety and efficacy...'
                }
            ]
        except Exception as e:
            st.error(f"Error fetching clinical trials: {str(e)}")
            return []

    def export_data(self, data: Dict, format: str = 'csv') -> Optional[bytes]:
        """Export analysis data in various formats"""
        try:
            # Prepare data for export
            export_dict = {
                'overview': data.get('overview', ''),
                'research_findings': [
                    {'title': f['title'], 'text': f['text']}
                    for f in data.get('pubmed_data', [])
                ],
                'clinical_trials': data.get('clinical_trials', [])
            }

            if format == 'csv':
                df = pd.json_normalize(export_dict)
                output = io.StringIO()
                df.to_csv(output, index=False)
                return output.getvalue().encode()
                
            elif format == 'json':
                return json.dumps(export_dict, indent=2).encode()
                
            elif format == 'pdf':
                # Placeholder for PDF export
                return None
                
        except Exception as e:
            st.error(f"Error exporting data: {str(e)}")
            return None

def main():
    st.set_page_config(page_title="Drug Research Assistant", page_icon="🧬", layout="wide")
    
    # Initialize session state
    if 'assistant' not in st.session_state:
        st.session_state.assistant = DrugResearchAssistant()
        
    # Create config instance
    config = Config()

    # Sidebar
    with st.sidebar:
        st.header("Analysis Options")
        drug_name = st.text_input("Drug Name", "imatinib")
        smiles = st.text_input("SMILES Notation (optional)", "")
        analysis_type = st.multiselect(
            "Analysis Types",
            config.ANALYSIS_TYPES,
            default=["mechanism of action"]
        )
        
        export_format = st.selectbox(
            "Export Format",
            ['csv', 'json', 'pdf']
        )
        
    # Main content
    st.title("🧬 Drug Research Assistant")
    st.write("Advanced drug analysis and research tool")
    
    if st.button("Analyze Drug"):
        # Create a progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Create tabs for different analysis sections
        tabs = st.tabs([
            "Overview",
            "Molecular Analysis",
            "Research Findings",
            "Clinical Trials",
            "Export"
        ])
        
        try:
            # Update progress - Initializing
            status_text.text("Initializing analysis...")
            progress_bar.progress(10)
            
            # Fetch and analyze data
            status_text.text("Fetching drug data from PubMed...")
            progress_bar.progress(20)
            
            # Create event loop and run async function
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            drug_data = loop.run_until_complete(
                st.session_state.assistant.fetch_drug_data(drug_name)
            )
            loop.close()
            
            progress_bar.progress(40)
            status_text.text("Processing research papers...")
            
            # Overview tab
            with tabs[0]:
                st.header("Drug Overview")
                status_text.text("Generating overview...")
                progress_bar.progress(50)
                st.write(drug_data.get('overview', 'No overview data available'))
                
            # Molecular Analysis tab
            with tabs[1]:
                status_text.text("Analyzing molecular properties...")
                progress_bar.progress(60)
                if smiles:
                    mol_analysis = st.session_state.assistant.analyze_molecular_properties(smiles)
                    if 'error' not in mol_analysis:
                        col1, col2 = st.columns([1, 1])
                        
                        with col1:
                            st.subheader("Molecular Properties")
                            for prop, value in mol_analysis['properties'].items():
                                st.write(f"{prop}: {value:.2f}" if isinstance(value, float) else f"{prop}: {value}")
                        
                        with col2:
                            st.subheader("Lipinski's Rule of Five")
                            st.info(st.session_state.assistant.explain_lipinski_rules())
                            for rule, passes in mol_analysis['lipinski'].items():
                                st.write(f"{rule}: {'✅' if passes else '❌'}")
                        
                        if mol_analysis.get('structure_image'):
                            st.image(
                                mol_analysis['structure_image'],
                                caption="Molecular Structure",
                                use_container_width=True
                            )
                    else:
                        st.error(mol_analysis['error'])
                else:
                    st.info("Enter SMILES notation for molecular analysis")
                    
            # Research Findings tab
            with tabs[2]:
                status_text.text("Analyzing research findings...")
                progress_bar.progress(75)
                st.header("Research Findings")
                summaries = st.session_state.assistant.summarize_research_findings(
                    drug_data.get('pubmed_data', [])
                )
                if isinstance(summaries, list):
                    for paper in summaries:
                        with st.expander(f"📄 {paper['title']}", expanded=False):
                            st.write(paper['summary'])
                else:
                    st.write(summaries)  # Show error message if summary failed
                
                # for finding in drug_data.get('pubmed_data', []):
                #     with st.expander(f"📄 {finding['title']}", expanded=False):
                #         st.write(finding['text'])
                    
            # Clinical Trials tab
            with tabs[3]:
                status_text.text("Processing clinical trials data...")
                progress_bar.progress(85)
                st.header("Clinical Trials")
                trials_df = pd.DataFrame(drug_data.get('clinical_trials', []))
                if not trials_df.empty:
                    st.dataframe(trials_df)
                else:
                    st.info("No clinical trials data available")
                    
            # Export tab
            with tabs[4]:
                status_text.text("Preparing export data...")
                progress_bar.progress(95)
                st.header("Export Data")
                export_data = st.session_state.assistant.export_data(drug_data, export_format)
                if export_data:
                    st.download_button(
                        label=f"Download {export_format.upper()}",
                        data=export_data,
                        file_name=f"{drug_name}_analysis.{export_format}",
                        mime=f"application/{export_format}"
                    )
            
            # Complete the progress bar
            progress_bar.progress(100)
            status_text.text("Analysis completed!")
            
            # Clear the progress indicators after a delay
            time.sleep(1)
            progress_bar.empty()
            status_text.empty()
            
        except Exception as e:
            progress_bar.empty()
            status_text.empty()
            st.error(f"Error during analysis: {str(e)}")

if __name__ == "__main__":
    main()