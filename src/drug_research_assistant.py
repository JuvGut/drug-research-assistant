# src/drug_research_assistant.py

import streamlit as st
import pandas as pd
import asyncio
import base64
from datetime import datetime
from typing import List, Dict, Optional
import io

from config import Config
from models import MoleculeAnalyzer
from utils import Cache, APIClient

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

    async def fetch_drug_data(self, drug_name: str) -> Dict:
        """Fetch comprehensive drug data from multiple sources"""
        try:
            # Check cache first
            cache_key = f"drug_data_{drug_name}"
            cached_data = self.cache.get(cache_key)
            if cached_data:
                return cached_data

            # Fetch data from multiple sources concurrently
            tasks = [
                self.fetch_pubmed_data(drug_name),
                self.fetch_clinical_trials(drug_name),
                self.fetch_drug_interactions(drug_name)
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Combine results
            drug_data = {
                'pubmed_data': results[0] if not isinstance(results[0], Exception) else [],
                'clinical_trials': results[1] if not isinstance(results[1], Exception) else [],
                'drug_interactions': results[2] if not isinstance(results[2], Exception) else []
            }
            
            # Cache the results
            self.cache.set(cache_key, drug_data)
            
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
            
            # Generate 2D structure image
            image_bytes = self.molecule_analyzer.generate_2d_image()
            image_b64 = base64.b64encode(image_bytes).decode() if image_bytes else None
            
            return {
                'properties': properties,
                'lipinski': lipinski,
                'structure_image': image_b64
            }
        except Exception as e:
            return {'error': str(e)}

    def generate_report(self, drug_name: str, analysis_data: Dict) -> str:
        """Generate a comprehensive analysis report"""
        try:
            report = f"# Drug Analysis Report: {drug_name}\n\n"
            report += f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            
            # Add molecular properties
            if 'molecular_properties' in analysis_data:
                report += "## Molecular Properties\n\n"
                props = analysis_data['molecular_properties']
                for key, value in props.items():
                    report += f"- {key.replace('_', ' ').title()}: {value}\n"
                    
            # Add research findings
            if 'research_findings' in analysis_data:
                report += "\n## Research Findings\n\n"
                findings = analysis_data['research_findings']
                for finding in findings:
                    report += f"### {finding['title']}\n{finding['text']}\n\n"
                    
            return report
        except Exception as e:
            return f"Error generating report: {str(e)}"

    def export_data(self, data: Dict, format: str = 'csv') -> Optional[bytes]:
        """Export analysis data in various formats"""
        try:
            if format == 'csv':
                df = pd.DataFrame(data)
                output = io.StringIO()
                df.to_csv(output, index=False)
                return output.getvalue().encode()
                
            elif format == 'json':
                return pd.DataFrame(data).to_json(orient='records').encode()
                
            elif format == 'pdf':
                # Implement PDF export using reportlab or another PDF library
                pass
                
        except Exception as e:
            st.error(f"Error exporting data: {str(e)}")
            return None

def main():
    st.set_page_config(page_title="Drug Research Assistant", page_icon="🧬", layout="wide")
    
    # Initialize session state
    if 'assistant' not in st.session_state:
        st.session_state.assistant = DrugResearchAssistant()
        
    # Sidebar
    with st.sidebar:
        st.header("Analysis Options")
        drug_name = st.text_input("Drug Name", "imatinib")
        smiles = st.text_input("SMILES Notation (optional)", "")
        analysis_type = st.multiselect(
            "Analysis Types",
            Config.ANALYSIS_TYPES,
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
        with st.spinner("Analyzing drug data..."):
            # Create tabs for different analysis sections
            tabs = st.tabs([
                "Overview",
                "Molecular Analysis",
                "Research Findings",
                "Clinical Trials",
                "Export"
            ])
            
            # Fetch and analyze data
            drug_data = asyncio.run(st.session_state.assistant.fetch_drug_data(drug_name))
            
            # Overview tab
            with tabs[0]:
                st.header("Drug Overview")
                st.write(drug_data.get('overview', 'No overview data available'))
                
            # Molecular Analysis tab
            with tabs[1]:
                if smiles:
                    mol_analysis = st.session_state.assistant.analyze_molecular_properties(smiles)
                    if 'error' not in mol_analysis:
                        st.subheader("Molecular Properties")
                        st.write(mol_analysis['properties'])
                        
                        st.subheader("Lipinski's Rule of Five")
                        st.write(mol_analysis['lipinski'])
                        
                        if mol_analysis.get('structure_image'):
                            st.image(mol_analysis['structure_image'])
                    else:
                        st.error(mol_analysis['error'])
                else:
                    st.info("Enter SMILES notation for molecular analysis")
                    
            # Research Findings tab
            with tabs[2]:
                st.header("Research Findings")
                for finding in drug_data.get('pubmed_data', []):
                    st.subheader(finding['title'])
                    st.write(finding['text'])
                    
            # Clinical Trials tab
            with tabs[3]:
                st.header("Clinical Trials")
                trials_df = pd.DataFrame(drug_data.get('clinical_trials', []))
                if not trials_df.empty:
                    st.dataframe(trials_df)
                else:
                    st.info("No clinical trials data available")
                    
            # Export tab
            with tabs[4]:
                st.header("Export Data")
                export_data = st.session_state.assistant.export_data(drug_data, export_format)
                if export_data:
                    st.download_button(
                        label=f"Download {export_format.upper()}",
                        data=export_data,
                        file_name=f"{drug_name}_analysis.{export_format}",
                        mime=f"application/{export_format}"
                    )

if __name__ == "__main__":
    main()