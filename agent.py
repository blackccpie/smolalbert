import os

from smolagents import CodeAgent, GradioUI, InferenceClientModel, Tool
from tavily import TavilyClient

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

# Set up the agent with the Tavily tool and a model
api_key = os.getenv("TAVILY_API_KEY")
search_tool = TavilySearchTool()
extract_tool = TavilyExtractTool()
model = InferenceClientModel(model_id="Qwen/Qwen3-Coder-30B-A3B-Instruct", provider="nebius")
agent = CodeAgent(
    tools=[search_tool, extract_tool], 
    model=model, 
    stream_outputs=True,
)

gradio_ui = GradioUI(agent, file_upload_folder=None, reset_agent_memory=True)
gradio_ui.launch(share=False)