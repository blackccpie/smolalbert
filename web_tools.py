# The MIT License

# Copyright (c) 2025 Albert Murienne

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import os
import requests

from smolagents import Tool
from tavily import TavilyClient

class TavilyBaseClient:
    __api_key = os.getenv("TAVILY_API_KEY")
    _tavily_client = TavilyClient(api_key=__api_key)

    @staticmethod
    def get_usage() -> str:
        url = "https://api.tavily.com/usage"
        headers = {
            "Authorization": f"Bearer {TavilyBaseClient.__api_key}",
            "Content-Type": "application/json",
        }
        res = requests.get(url, headers=headers)
        res.raise_for_status()
        
        account = res.json().get("account", {})
        plan_usage = account.get("plan_usage")
        plan_limit = account.get("plan_limit")

        return f"{plan_usage}/{plan_limit}"

class TavilySearchTool(TavilyBaseClient, Tool):
    """
    A tool to perform web searches using the Tavily API.
    """
    name = "tavily_search"
    description = "Search the web using Tavily."
    inputs = {
        "query": {
            "type": "string",
            "description": "The search query string.",
        }
    }
    output_type = "string"

    def forward(self, query: str):
        response = self._tavily_client.search(query)
        return response

class TavilyExtractTool(TavilyBaseClient, Tool):
    """
    A tool to extract information from web pages using the Tavily API.
    """
    name = "tavily_extract"
    description = "Extract information from web pages using Tavily."
    inputs = {
        "url": {
            "type": "string",
            "description": "The URL of the web page to extract information from.",
        }
    }
    output_type = "string"

    def forward(self, url: str):
        response = self._tavily_client.extract(url)
        return response

class TavilyImageURLSearchTool(TavilyBaseClient, Tool):
    """
    A tool to search for image URLs using the Tavily API.
    """
    name = "tavily_image_search"
    description = "Search for most relevant image URL on the web using Tavily."
    inputs = {
        "query": {
            "type": "string",
            "description": "The search query string.",
        }
    }
    output_type = "string"

    def forward(self, query: str):
        response = self._tavily_client.search(query, include_images=True)
        
        images = response.get("images", [])
        
        if images:
            # Return the URL of the first image
            first_image = images[0]
            if isinstance(first_image, dict):
                return first_image.get("url")
            return first_image
        return "none"