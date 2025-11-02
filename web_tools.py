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

# ---------------------------------------------------------------------
# Tavily Tools
# ---------------------------------------------------------------------

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

    # Consumes 1 Tavily credit per query
    __basic_params = {
            "search_depth": "basic",
            "max_results": 10,              # fetch up to 10 sources
            "auto_parameters": False,       # keep manual control
            "include_raw_content": False,
        }

    # Consumes 2 Tavily credits per query
    __advanced_params = {
            "search_depth": "advanced",     # 'advanced' yields better relevance
            "max_results": 10,              # fetch up to 10 sources
            "chunks_per_source": 3,         # number of content snippets to return per source
            "auto_parameters": False,       # keep manual control
            "include_raw_content": False,
        }

    def __init__(self):
        """
        Construct the TavilySearchTool.
        """
        # Call superclass constructor
        super().__init__()


        self.params = TavilySearchTool.__basic_params

    def enable_advanced_mode(self, enable: bool = True):
        """
        Enable or disable advanced mode for the search tool.
        Advanced mode uses more credits but yields better results.
        """
        if enable:
            self.params = TavilySearchTool.__advanced_params
        else:
            self.params = TavilySearchTool.__basic_params
        
        print(f"TavilySearchTool advanced mode has been {'enabled' if enable else 'disabled'}.")

    def forward(self, query: str):

        params = self.params
        params["query"] = query

        try:
            response = self._tavily_client.search(**params)
        except Exception as e:
            return f"Error calling Tavily API: {e}"
        
        return response

class TavilyExtractTool(TavilyBaseClient, Tool):
    """
    A tool to extract raw information from web pages using the Tavily API.
    """
    name = "tavily_extract"
    description = "Extract raw information from web pages using Tavily."
    inputs = {
        "url": {
            "type": "string",
            "description": "The URL of the web page to extract information from.",
        }
    }
    output_type = "string"

    def __init__(self):
        """
        Construct the TavilyExtractTool.
        """
        # Call superclass constructor
        super().__init__()

        self.extract_depth = "basic"

    def enable_advanced_mode(self, enable: bool = True):
        """
        Enable or disable advanced mode for the extract tool.
        Advanced mode uses more credits but yields better results (retrieves more data, including tables and embedded content).
        """
        if enable:
            self.extract_depth = "advanced"
        else:
            self.extract_depth = "basic"

        print(f"TavilyExtractTool advanced mode has been {'enabled' if enable else 'disabled'}.")

    def forward(self, url: str):
        try:
            response = self._tavily_client.extract(
                urls=url,
                extract_depth=self.extract_depth)
        except Exception as e:
            return f"Error calling Tavily extract API: {e}"

        # Tavily's Extract API can return raw HTML + text.
        # you may trim or sanitize here if needed.
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
        try:
            response = self._tavily_client.search(
                query,
                include_images=True,
                include_image_descriptions=True,
                max_results=5
            )
        except Exception as e:
            return f"Error calling Tavily API: {e}"

        images = response.get("images", [])
        if not images:
            return "none"

        # Return the URL of the first image
        first_image = images[0]
        if isinstance(first_image, dict):
            return first_image.get("url", "none")
        return first_image or "none"
