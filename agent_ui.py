# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
import re
import shutil
from pathlib import Path
from typing import Generator

from smolagents.agent_types import AgentAudio, AgentImage, AgentText
from smolagents.agents import MultiStepAgent, PlanningStep
from smolagents.memory import ActionStep, FinalAnswerStep
from smolagents.models import ChatMessageStreamDelta, MessageRole, agglomerate_stream_deltas
from smolagents.utils import _is_package_available


def get_step_footnote_content(step_log: ActionStep | PlanningStep, step_name: str) -> str:
    """Get a footnote string for a step log with duration and token information"""
    step_footnote = f"**{step_name}**"
    if step_log.token_usage is not None:
        step_footnote += f" | Input tokens: {step_log.token_usage.input_tokens:,} | Output tokens: {step_log.token_usage.output_tokens:,}"
    step_footnote += f" | Duration: {round(float(step_log.timing.duration), 2)}s" if step_log.timing.duration else ""
    step_footnote_content = f"""<span style="color: #bbbbc2; font-size: 12px;">{step_footnote}</span> """
    return step_footnote_content


def _clean_model_output(model_output: str) -> str:
    """
    Clean up model output by removing trailing tags and extra backticks.

    Args:
        model_output (`str`): Raw model output.

    Returns:
        `str`: Cleaned model output.
    """
    if not model_output:
        return ""
    model_output = model_output.strip()
    # Remove any trailing <end_code> and extra backticks, handling multiple possible formats
    model_output = re.sub(r"```\s*<end_code>", "```", model_output)  # handles ```<end_code>
    model_output = re.sub(r"<end_code>\s*```", "```", model_output)  # handles <end_code>```
    model_output = re.sub(r"```\s*\n\s*<end_code>", "```", model_output)  # handles ```\n<end_code>
    return model_output.strip()


def _format_code_content(content: str) -> str:
    """
    Format code content as Python code block if it's not already formatted.

    Args:
        content (`str`): Code content to format.

    Returns:
        `str`: Code content formatted as a Python code block.
    """
    content = content.strip()
    # Remove existing code blocks and end_code tags
    content = re.sub(r"```.*?\n", "", content)
    content = re.sub(r"\s*<end_code>\s*", "", content)
    content = content.strip()
    # Add Python code block formatting if not already present
    if not content.startswith("```python"):
        content = f"```python\n{content}\n```"
    return content


def _process_action_step(step_log: ActionStep, skip_model_outputs: bool = False) -> Generator:
    """
    Process an [`ActionStep`] and yield appropriate Gradio ChatMessage objects.

    Args:
        step_log ([`ActionStep`]): ActionStep to process.
        skip_model_outputs (`bool`): Whether to skip model outputs.

    Yields:
        `gradio.ChatMessage`: Gradio ChatMessages representing the action step.
    """
    import gradio as gr

    # Output the step number
    step_number = f"Step {step_log.step_number}"
    if not skip_model_outputs:
        yield gr.ChatMessage(role=MessageRole.ASSISTANT, content=f"**{step_number}**", metadata={"status": "done"})

    # First yield the thought/reasoning from the LLM
    if not skip_model_outputs and getattr(step_log, "model_output", ""):
        model_output = _clean_model_output(step_log.model_output)
        yield gr.ChatMessage(role=MessageRole.ASSISTANT, content=model_output, metadata={"status": "done"})

    # For tool calls, create a parent message
    if getattr(step_log, "tool_calls", []):
        first_tool_call = step_log.tool_calls[0]
        used_code = first_tool_call.name == "python_interpreter"

        # Process arguments based on type
        args = first_tool_call.arguments
        if isinstance(args, dict):
            content = str(args.get("answer", str(args)))
        else:
            content = str(args).strip()

        # Format code content if needed
        if used_code:
            content = _format_code_content(content)

        # Create the tool call message
        parent_message_tool = gr.ChatMessage(
            role=MessageRole.ASSISTANT,
            content=content,
            metadata={
                "title": f"🛠️ Used tool {first_tool_call.name}",
                "status": "done",
            },
        )
        yield parent_message_tool

    # Display execution logs if they exist
    if getattr(step_log, "observations", "") and step_log.observations.strip():
        log_content = step_log.observations.strip()
        if log_content:
            log_content = re.sub(r"^Execution logs:\s*", "", log_content)
            yield gr.ChatMessage(
                role=MessageRole.ASSISTANT,
                content=f"```bash\n{log_content}\n",
                metadata={"title": "📝 Execution Logs", "status": "done"},
            )

    # Display any images in observations
    if getattr(step_log, "observations_images", []):
        for image in step_log.observations_images:
            path_image = AgentImage(image).to_string()
            yield gr.ChatMessage(
                role=MessageRole.ASSISTANT,
                content={"path": path_image, "mime_type": f"image/{path_image.split('.')[-1]}"},
                metadata={"title": "🖼️ Output Image", "status": "done"},
            )

    # Handle errors
    if getattr(step_log, "error", None):
        yield gr.ChatMessage(
            role=MessageRole.ASSISTANT, content=str(step_log.error), metadata={"title": "💥 Error", "status": "done"}
        )

    # Add step footnote and separator
    yield gr.ChatMessage(
        role=MessageRole.ASSISTANT,
        content=get_step_footnote_content(step_log, step_number),
        metadata={"status": "done"},
    )
    yield gr.ChatMessage(role=MessageRole.ASSISTANT, content="-----", metadata={"status": "done"})


def _process_planning_step(step_log: PlanningStep, skip_model_outputs: bool = False) -> Generator:
    """
    Process a [`PlanningStep`] and yield appropriate gradio.ChatMessage objects.

    Args:
        step_log ([`PlanningStep`]): PlanningStep to process.

    Yields:
        `gradio.ChatMessage`: Gradio ChatMessages representing the planning step.
    """
    import gradio as gr

    if not skip_model_outputs:
        yield gr.ChatMessage(role=MessageRole.ASSISTANT, content="**Planning step**", metadata={"status": "done"})
        yield gr.ChatMessage(role=MessageRole.ASSISTANT, content=step_log.plan, metadata={"status": "done"})
    yield gr.ChatMessage(
        role=MessageRole.ASSISTANT,
        content=get_step_footnote_content(step_log, "Planning step"),
        metadata={"status": "done"},
    )
    yield gr.ChatMessage(role=MessageRole.ASSISTANT, content="-----", metadata={"status": "done"})


def _process_final_answer_step(step_log: FinalAnswerStep) -> Generator:
    """
    Process a [`FinalAnswerStep`] and yield appropriate gradio.ChatMessage objects.

    Args:
        step_log ([`FinalAnswerStep`]): FinalAnswerStep to process.

    Yields:
        `gradio.ChatMessage`: Gradio ChatMessages representing the final answer.
    """
    import gradio as gr

    final_answer = step_log.output
    if isinstance(final_answer, AgentText):
        yield gr.ChatMessage(
            role=MessageRole.ASSISTANT,
            content=f"**Final answer:**\n{final_answer.to_string()}\n",
            metadata={"status": "done"},
        )
    elif isinstance(final_answer, AgentImage):
        yield gr.ChatMessage(
            role=MessageRole.ASSISTANT,
            content={"path": final_answer.to_string(), "mime_type": "image/png"},
            metadata={"status": "done"},
        )
    elif isinstance(final_answer, AgentAudio):
        yield gr.ChatMessage(
            role=MessageRole.ASSISTANT,
            content={"path": final_answer.to_string(), "mime_type": "audio/wav"},
            metadata={"status": "done"},
        )
    else:
        yield gr.ChatMessage(
            role=MessageRole.ASSISTANT, content=f"**Final answer:** {str(final_answer)}", metadata={"status": "done"}
        )


def pull_messages_from_step(step_log: ActionStep | PlanningStep | FinalAnswerStep, skip_model_outputs: bool = False):
    """Extract Gradio ChatMessage objects from agent steps with proper nesting.

    Args:
        step_log: The step log to display as gr.ChatMessage objects.
        skip_model_outputs: If True, skip the model outputs when creating the gr.ChatMessage objects:
            This is used for instance when streaming model outputs have already been displayed.
    """
    if isinstance(step_log, ActionStep):
        yield from _process_action_step(step_log, skip_model_outputs)
    elif isinstance(step_log, PlanningStep):
        yield from _process_planning_step(step_log, skip_model_outputs)
    elif isinstance(step_log, FinalAnswerStep):
        yield from _process_final_answer_step(step_log)
    else:
        raise ValueError(f"Unsupported step type: {type(step_log)}")


def stream_to_gradio(
    agent,
    task: str,
    task_images: list | None = None,
    reset_agent_memory: bool = False,
    additional_args: dict | None = None,
) -> Generator:
    """Runs an agent with the given task and streams the messages from the agent as gradio ChatMessages."""

    accumulated_events: list[ChatMessageStreamDelta] = []
    for event in agent.run(
        task, images=task_images, stream=True, reset=reset_agent_memory, additional_args=additional_args
    ):
        if isinstance(event, ActionStep | PlanningStep | FinalAnswerStep):
            for message in pull_messages_from_step(
                event,
                # If we're streaming model outputs, no need to display them twice
                skip_model_outputs=getattr(agent, "stream_outputs", False),
            ):
                yield message
            accumulated_events = []
        elif isinstance(event, ChatMessageStreamDelta):
            accumulated_events.append(event)
            text = agglomerate_stream_deltas(accumulated_events).render_as_markdown()
            yield text


class AgentUI:
    """
    Gradio interface for interacting with a [`MultiStepAgent`].

    This class provides a web interface to interact with the agent in real-time, allowing users to submit prompts, and receive responses in a chat-like format.
    It  can reset the agent's memory at the start of each interaction if desired.
    It uses the [`gradio.Chatbot`] component to display the conversation history.
    This class requires the `gradio` extra to be installed: `pip install 'smolagents[gradio]'`.

    Args:
        agent ([`MultiStepAgent`]): The agent to interact with.
        reset_agent_memory (`bool`, *optional*, defaults to `False`): Whether to reset the agent's memory at the start of each interaction.
            If `True`, the agent will not remember previous interactions.
    """

    def __init__(self, agent: MultiStepAgent, reset_agent_memory: bool = False):
        self.agent = agent
        self.reset_agent_memory = reset_agent_memory
        self.description = getattr(agent, "description", None)

    def interact_with_agent(self, prompt, verbose_messages, quiet_messages, session_state):
        """
        Interacts with the agent and streams results into two separate histories:
            - verbose_messages: full reasoning stream (Chatterbox)
            - quiet_messages: only user prompt + final answer (Quiet)
        Quiet is enhanced with pending "Step N..." indicators only (no generic thinking text).
        """
        import gradio as gr

        # Get the agent type from the template agent
        if "agent" not in session_state:
            session_state["agent"] = self.agent

        # clear previous messages
        verbose_messages = []
        quiet_messages = []

        try:
            # Append the user message to both histories (quiet keeps the user query)
            user_msg = gr.ChatMessage(role="user", content=prompt, metadata={"status": "done"})
            verbose_messages.append(user_msg)
            quiet_messages.append(user_msg)

            # yield initial state to update UI immediately
            yield verbose_messages, quiet_messages

            quiet_pending_idx = None

            for msg in stream_to_gradio(
                session_state["agent"], task=prompt, reset_agent_memory=self.reset_agent_memory
            ):

                # Full gr.ChatMessage object (from steps) — append to verbose always
                if isinstance(msg, gr.ChatMessage):
                    # Mark last verbose pending -> done if needed and append
                    if verbose_messages and verbose_messages[-1].metadata.get("status") == "pending":
                        verbose_messages[-1].metadata["status"] = "done"
                        verbose_messages[-1].content = msg.content
                    else:
                        verbose_messages.append(msg)

                    content_text = msg.content if isinstance(msg.content, str) else ""

                    # Detect final answer messages and append to quiet
                    # HACK : FinalAnswerStep messages are produced by _process_final_answer_step and use "**Final answer:**" text
                    if "final answer" in content_text.lower():
                        # Replace pending with final answer in Quiet
                        final_msg = gr.ChatMessage(role=MessageRole.ASSISTANT, content=content_text, metadata={"status": "done"})
                        if quiet_pending_idx is not None:
                            quiet_messages[quiet_pending_idx] = final_msg
                            quiet_pending_idx = None
                        else:
                            quiet_messages.append(final_msg)
                    else:
                        # Look for "Step <number>" pattern
                        match = re.search(r"\bStep\s*(\d+)\b", content_text, re.IGNORECASE)
                        if match:
                            step_num = match.group(1)
                            pending_text = f"⏳ Step {step_num}..."
                            if quiet_pending_idx is None:
                                quiet_messages.append(
                                    gr.ChatMessage(
                                        role=MessageRole.ASSISTANT,
                                        content=pending_text,
                                        metadata={"status": "pending"},
                                    )
                                )
                                quiet_pending_idx = len(quiet_messages) - 1
                            else:
                                quiet_messages[quiet_pending_idx].content = pending_text

                elif isinstance(msg, str):
                    text = msg.replace("<", r"\<").replace(">", r"\>")
                    if verbose_messages and verbose_messages[-1].metadata.get("status") == "pending":
                        verbose_messages[-1].content = text
                    else:
                        verbose_messages.append(
                            gr.ChatMessage(role=MessageRole.ASSISTANT, content=text, metadata={"status": "pending"})
                        )
                    yield verbose_messages, quiet_messages

            # final yield to ensure both UIs are up-to-date
            yield verbose_messages, quiet_messages

        except Exception as e:
            # ensure UIs don't hang if something failed
            yield verbose_messages, quiet_messages
            raise gr.Error(f"Error in interaction: {str(e)}")

    def launch(self, share: bool = True, **kwargs):
        """
        Launch the Gradio app with the agent interface.

        Args:
            share (`bool`, defaults to `True`): Whether to share the app publicly.
            **kwargs: Additional keyword arguments to pass to the Gradio launch method.
        """
        self.create_app().launch(debug=True, share=share, **kwargs)

    def create_app(self):
        import gradio as gr

        with gr.Blocks(theme="glass", fill_height=True) as agent:

            # Add session state to store session-specific data
            session_state = gr.State({})
            stored_messages_verbose = gr.State([])  # full reasoning history
            stored_messages_quiet = gr.State([])    # only user + final answer

            with gr.Sidebar():
                gr.Markdown(
                    "# SmolAlbert 🤖"
                )

                with gr.Group():
                    gr.Markdown("**Your request**", container=True)
                    text_input = gr.Textbox(
                        lines=3,
                        label="Chat Message",
                        container=False,
                        placeholder="Enter your prompt here and press Shift+Enter or press the button",
                    )
                    submit_btn = gr.Button("Submit", variant="primary")

                gr.HTML(
                    "<br><br><h4><center>Powered by <a target='_blank' href='https://github.com/huggingface/smolagents'><b>smolagents</b></a></center></h4>"
                )

            with gr.Tab("Quiet"):
                quiet_chatbot = gr.Chatbot(
                        label="Agent",
                        type="messages",
                        avatar_images=(
                            None,
                            "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/smolagents/mascot_smol.png",
                        ),
                        resizeable=True,
                        scale=1,
                        latex_delimiters=[
                            {"left": r"$$", "right": r"$$", "display": True},
                            {"left": r"$", "right": r"$", "display": False},
                            {"left": r"\[", "right": r"\]", "display": True},
                            {"left": r"\(", "right": r"\)", "display": False},
                        ],
                    )

            with gr.Tab("Chatterbox"):

                # Main chat interface
                verbose_chatbot = gr.Chatbot(
                    label="Agent",
                    type="messages",
                    avatar_images=(
                        None,
                        "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/smolagents/mascot_smol.png",
                    ),
                    resizeable=True,
                    scale=1,
                    latex_delimiters=[
                        {"left": r"$$", "right": r"$$", "display": True},
                        {"left": r"$", "right": r"$", "display": False},
                        {"left": r"\[", "right": r"\]", "display": True},
                        {"left": r"\(", "right": r"\)", "display": False},
                    ],
                )

            # Main input handlers: call interact_with_agent(prompt, verbose_state, quiet_state, session_state)
            text_input.submit(
                self.interact_with_agent,
                [text_input, stored_messages_verbose, stored_messages_quiet, session_state],
                [verbose_chatbot, quiet_chatbot],
            ).then(
                lambda: (
                    gr.Textbox(
                        interactive=True, placeholder="Enter your prompt here and press Shift+Enter or the button"
                    ),
                    gr.Button(interactive=True),
                ),
                None,
                [text_input, submit_btn],
            )

            submit_btn.click(
                self.interact_with_agent,
                [text_input, stored_messages_verbose, stored_messages_quiet, session_state],
                [verbose_chatbot, quiet_chatbot],
            ).then(
                lambda: (
                    gr.Textbox(
                        interactive=True, placeholder="Enter your prompt here and press Shift+Enter or the button"
                    ),
                    gr.Button(interactive=True),
                ),
                None,
                [text_input, submit_btn],
            )

            # bind clears to both chat components so agent memory is reset
            quiet_chatbot.clear(self.agent.memory.reset)
            verbose_chatbot.clear(self.agent.memory.reset)

        return agent


__all__ = ["stream_to_gradio", "AgentUI"]
