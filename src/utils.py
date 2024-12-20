# src/utils.py

import aiohttp
import asyncio
from typing import Dict, Any, Optional, Union
import json
import os
import time
import hashlib
from bs4 import BeautifulSoup
import xml.etree.ElementTree as ET

class APIClient:
    def __init__(self, base_url: str, api_key: str = None):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.session = None
        
    async def _get_session(self):
        if self.session is None:
            self.session = aiohttp.ClientSession()
        return self.session
        
    def _build_url(self, endpoint: str) -> str:
        """Build full URL for API endpoint"""
        return f"{self.base_url}/{endpoint.lstrip('/')}"
        
    def _add_api_key(self, params: Dict) -> Dict:
        """Add API key to parameters if available"""
        if self.api_key:
            params['api_key'] = self.api_key
        return params
        
    async def get(self, endpoint: str, params: Dict = None, response_format: str = 'json') -> Union[Dict, str]:
        """Make GET request to API endpoint"""
        if params is None:
            params = {}
            
        params = self._add_api_key(params)
        url = self._build_url(endpoint)
        
        session = await self._get_session()
        try:
            async with session.get(url, params=params, timeout=10) as response:
                response.raise_for_status()
                
                if response_format == 'json':
                    return await response.json()
                elif response_format == 'xml':
                    return await response.text()
                else:
                    return await response.text()
                    
        except Exception as e:
            raise Exception(f"API request failed: {str(e)}")

    async def close(self):
        """Close the aiohttp session"""
        if self.session is not None:
            await self.session.close()
            self.session = None

class Cache:
    def __init__(self, cache_dir: str, expiry: int):
        self.cache_dir = cache_dir
        self.expiry = expiry
        os.makedirs(cache_dir, exist_ok=True)
        
    def _get_cache_path(self, key: str) -> str:
        """Generate cache file path from key"""
        hashed_key = hashlib.md5(key.encode()).hexdigest()
        return os.path.join(self.cache_dir, f"{hashed_key}.json")
        
    def get(self, key: str) -> Optional[Dict]:
        """Retrieve data from cache if not expired"""
        cache_path = self._get_cache_path(key)
        
        if not os.path.exists(cache_path):
            return None
            
        try:
            with open(cache_path, 'r') as f:
                data = json.load(f)
                
            if time.time() - data['timestamp'] > self.expiry:
                os.remove(cache_path)
                return None
                
            return data['content']
        except Exception:
            return None
            
    def set(self, key: str, content: Dict) -> None:
        """Save data to cache"""
        cache_path = self._get_cache_path(key)
        
        try:
            cache_data = {
                'timestamp': time.time(),
                'content': content
            }
            
            with open(cache_path, 'w') as f:
                json.dump(cache_data, f)
        except Exception:
            pass