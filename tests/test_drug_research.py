# tests/test_drug_research.py

import pytest
from src.models import MoleculeAnalyzer
from src.utils import Cache, APIClient
import os
import tempfile
import json

@pytest.fixture
def molecule_analyzer():
    return MoleculeAnalyzer()

@pytest.fixture
def cache():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Cache(tmpdir, expiry=60)

@pytest.fixture
def api_client():
    return APIClient("https://api.example.com", "test_key")

class TestMoleculeAnalyzer:
    def test_load_smiles_valid(self, molecule_analyzer):
        assert molecule_analyzer.load_smiles("CC(=O)OC1=CC=CC=C1C(=O)O")
        
    def test_load_smiles_invalid(self, molecule_analyzer):
        assert not molecule_analyzer.load_smiles("invalid_smiles")
        
    def test_calculate_properties(self, molecule_analyzer):
        molecule_analyzer.load_smiles("CC(=O)OC1=CC=CC=C1C(=O)O")
        props = molecule_analyzer.calculate_properties()
        assert isinstance(props, dict)
        assert 'molecular_weight' in props
        assert 'logp' in props
        
    def test_check_lipinski(self, molecule_analyzer):
        molecule_analyzer.load_smiles("CC(=O)OC1=CC=CC=C1C(=O)O")
        lipinski = molecule_analyzer.check_lipinski()
        assert isinstance(lipinski, dict)
        assert 'passes_all' in lipinski

class TestCache:
    def test_set_and_get(self, cache):
        data = {"test": "data"}
        cache.set("test_key", data)
        assert cache.get("test_key") == data
        
    def test_expired_data(self, cache):
        cache.expiry = 0  # Immediate expiration
        cache.set("test_key", {"test": "data"})
        assert cache.get("test_key") is None
        
    def test_invalid_data(self, cache):
        assert cache.get("nonexistent_key") is None

class TestAPIClient:
    def test_build_url(self, api_client):
        assert api_client._build_url("/test") == "https://api.example.com/test"
        
    def test_add_api_key(self, api_client):
        params = {}
        params_with_key = api_client._add_api_key(params)
        assert params_with_key['api_key'] == "test_key"