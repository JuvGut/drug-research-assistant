# src/utils.py

import aiohttp
import asyncio
from typing import Dict, Any, Optional, Union, List
import json
import os
import time
import hashlib
from bs4 import BeautifulSoup
import xml.etree.ElementTree as ET

class APIClient:
    def __init__(self, base_url: str, api_key: str = None):
        self.base_url = base_url.rstrip('/')
        if not api_key:
            raise ValueError("API key is required but not provided. Please check your .env file.")
        self.api_key = api_key
        self._session = None
        self._connector = None
        
    async def _get_connector(self):
        if self._connector is None:
            self._connector = aiohttp.TCPConnector(
                limit=10,
                ttl_dns_cache=300,
                use_dns_cache=True
            )
        return self._connector
        
    async def _get_session(self):
        if self._session is None or self._session.closed:
            connector = await self._get_connector()
            self._session = aiohttp.ClientSession(
                connector=connector,
                timeout=aiohttp.ClientTimeout(total=30),
                headers={'Connection': 'keep-alive'}
            )
        return self._session

    def _build_url(self, endpoint: str) -> str:
        return f"{self.base_url}/{endpoint.lstrip('/')}"
        
    def _add_api_key(self, params: Dict) -> Dict:
        if not self.api_key:
            raise ValueError("API key is not configured. Please check your .env file.")
        params['api_key'] = self.api_key
        return params

    async def batch_get(self, requests: List[Dict]) -> List[Dict]:
        """Execute multiple requests concurrently"""
        session = await self._get_session()
        async with session:
            tasks = []
            for req in requests:
                endpoint = req.get('endpoint', '')
                params = req.get('params', {})
                params = self._add_api_key(params)
                url = self._build_url(endpoint)
                
                tasks.append(self._single_request(session, url, params))
            
            return await asyncio.gather(*tasks, return_exceptions=True)
            
    async def _single_request(self, session, url: str, params: Dict) -> Dict:
        """Execute a single request with retry logic"""
        retries = 3
        delay = 1
        
        for attempt in range(retries):
            try:
                async with session.get(url, params=params) as response:
                    response.raise_for_status()
                    return await response.json()
            except Exception as e:
                if attempt == retries - 1:
                    raise
                await asyncio.sleep(delay * (2 ** attempt))

    async def get(self, endpoint: str, params: Dict = None, response_format: str = 'json') -> Union[Dict, str]:
        """Make GET request to API endpoint"""
        if params is None:
            params = {}
            
        params = self._add_api_key(params)
        url = self._build_url(endpoint)
        
        try:
            current_session = await self._get_session()
            async with current_session.get(url, params=params, timeout=30) as response:
                try:
                    response.raise_for_status()
                    
                    if response_format == 'json':
                        return await response.json()
                    elif response_format == 'xml':
                        return await response.text()
                    else:
                        return await response.text()
                except aiohttp.ContentTypeError:
                    text = await response.text()
                    if response_format == 'json':
                        try:
                            return json.loads(text)
                        except json.JSONDecodeError:
                            raise Exception(f"Invalid JSON response: {text[:200]}...")
                    return text
                    
        except Exception as e:
            raise Exception(f"API request failed: {str(e)}")

    async def close(self):
        if self._session is not None and not self._session.closed:
            await self._session.close()
            self._session = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
        
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