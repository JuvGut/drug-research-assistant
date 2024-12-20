# src/models.py

from rdkit import Chem
from rdkit.Chem import AllChem, Draw
import numpy as np
from typing import Optional, List, Dict
import io

class MoleculeAnalyzer:
    def __init__(self):
        self.mol = None
        
    def load_smiles(self, smiles: str) -> bool:
        """Load molecule from SMILES string"""
        try:
            self.mol = Chem.MolFromSmiles(smiles)
            return self.mol is not None
        except Exception:
            return False
            
    def generate_2d_image(self) -> Optional[bytes]:
        """Generate 2D molecule image"""
        if self.mol is None:
            return None
            
        try:
            img = Draw.MolToImage(self.mol)
            img_bytes = io.BytesIO()
            img.save(img_bytes, format='PNG')
            return img_bytes.getvalue()
        except Exception:
            return None
            
    def calculate_properties(self) -> Dict:
        """Calculate molecular properties"""
        if self.mol is None:
            return {}
            
        try:
            return {
                'molecular_weight': Chem.Descriptors.ExactMolWt(self.mol),
                'logp': Chem.Descriptors.MolLogP(self.mol),
                'hbd': Chem.Descriptors.NumHDonors(self.mol),
                'hba': Chem.Descriptors.NumHAcceptors(self.mol),
                'tpsa': Chem.Descriptors.TPSA(self.mol),
                'rotatable_bonds': Chem.Descriptors.NumRotatableBonds(self.mol)
            }
        except Exception:
            return {}
            
    def check_lipinski(self) -> Dict[str, bool]:
        """Check Lipinski's Rule of Five"""
        props = self.calculate_properties()
        if not props:
            return {}
            
        return {
            'molecular_weight_ok': props['molecular_weight'] <= 500,
            'logp_ok': abs(props['logp']) <= 5,
            'hbd_ok': props['hbd'] <= 5,
            'hba_ok': props['hba'] <= 10,
            'passes_all': all([
                props['molecular_weight'] <= 500,
                abs(props['logp']) <= 5,
                props['hbd'] <= 5,
                props['hba'] <= 10
            ])
        }