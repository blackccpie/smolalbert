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

from smolagents import CodeAgent, InferenceClientModel, Tool
from tavily import TavilyClient

from agent_ui import AgentUI

# Define a custom tool for Tavily search
from smolagents import Tool
from tavily import TavilyClient

tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

class TavilySearchTool(Tool):
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
        response = tavily_client.search(query)
        return {"results": response}
    
class TavilyExtractTool(Tool):
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
        response = tavily_client.extract(url)
        return {"results": response}

#model_id = "Qwen/Qwen3-Coder-30B-A3B-Instruct"
#model_id = "Qwen/Qwen3-30B-A3B-Thinking-2507"
model_id = "Qwen/Qwen3-235B-A22B-Instruct-2507"
provider = "nebius"

# Set up the agent with the Tavily tool and a model
api_key = os.getenv("TAVILY_API_KEY")
search_tool = TavilySearchTool()
extract_tool = TavilyExtractTool()
model = InferenceClientModel(model_id=model_id, provider=provider)
agent = CodeAgent(
    tools=[search_tool, extract_tool], 
    model=model, 
    stream_outputs=True,
    instructions=(
        "When writing the final answer, including the most relevant URL(s) "
        "from your search results as inline Markdown hyperlinks is MANDATORY. "
        "Example format: ... (see [1](https://example1.com)) ... (see [2](https://example2.com)) ... "
        "Do not invent URL(s) — only use the ones you were provided."
    )
)

agent_ui = AgentUI(agent, reset_agent_memory=True)
agent_ui.launch(share=False)