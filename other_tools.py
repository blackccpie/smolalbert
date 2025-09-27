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
from pyexpat.errors import messages

from smolagents import InferenceClientModel, Tool

class ImageQueryTool(Tool):
    """
    A tool to ask a question about an image given its URL.
    """
    name = "image_query"
    description = "Ask a question about an image given its URL."
    inputs = {
        "image_url": {
            "type": "string",
            "description": "The URL of the image to analyze.",
        },
        "question": {
            "type": "string",
            "description": "The question to ask about the image.",
        }
    }
    output_type = "string"

    def __init__(self):
        """
        Construct the ImageQueryTool with a specific model.
        """
        # call superclass constructor
        super().__init__()
        # Initialize the model
        self.model = InferenceClientModel(
            model_id="google/gemma-3-27b-it",
            provider="auto",
            token=os.getenv("HF_API_KEY")
        )

    def forward(self, image_url: str, question: str):
        """
        Forward method to process the image URL and question.
        """
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {"type": "image_url", "image_url": {"url": image_url}}
                ],
            }
        ]

        res = self.model(messages)
        return res.content

if __name__ == "__main__":
    tool = ImageQueryTool()
    response = tool.forward(
        image_url="https://upload.wikimedia.org/wikipedia/commons/9/99/Black_square.jpg",
        question="What is this?"
    )
    print("Response:", response)