# ipython llm.python.py
import logging
from autogen import ConversableAgent, GroupChat, GroupChatManager, Agent
from autogen.coding import CodeExecutor, CodeBlock, CodeResult, CodeExtractor, MarkdownCodeExtractor
from typing import List, Union, Literal
from IPython import get_ipython

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# LLM Config
llm_config = {
    "config_list": [
        {
            "model": "NotRequired",  # Loaded with LiteLLM command
            "api_key": "NotRequired",  # Not needed
            "base_url": "http://0.0.0.0:4000",  # Your LiteLLM URL
            "price": [0, 0],  # Put in price per 1K tokens [prompt, response] as free!
        }
    ],
    "cache_seed": None,  # Turns off caching, useful for testing different models
}

# Custom Code Executor
class NotebookExecutor(CodeExecutor):

    @property
    def code_extractor(self) -> CodeExtractor:
        # Extract code from markdown blocks.
        return MarkdownCodeExtractor()

    def __init__(self) -> None:
        # Get the current IPython instance running in this notebook.
        self._ipython = get_ipython()
        if self._ipython is None:
            raise RuntimeError("No active IPython instance found. Please ensure this code is run in an IPython environment.")

    def execute_code_blocks(self, code_blocks: List[CodeBlock]) -> CodeResult:
        log = ""
        exit_code = 0  # Initialize exit code
        for code_block in code_blocks:
            result = self._ipython.run_cell("%%capture --no-display cap\n" + code_block.code)
            log += self._ipython.ev("cap.stdout")
            log += self._ipython.ev("cap.stderr")
            if result.result is not None:
                log += str(result.result)
            exit_code = 0 if result.success else 1
            if result.error_before_exec is not None:
                log += f"\n{result.error_before_exec}"
                exit_code = 1
            if result.error_in_exec is not None:
                log += f"\n{result.error_in_exec}"
                exit_code = 1
            if exit_code != 0:
                break
        return CodeResult(exit_code=exit_code, output=log)

# Generic Manager Agent
manager_agent = ConversableAgent(
    name="manager_agent",
    system_message="""You are the Manager Agent. Your task is to identify tasks and set code requirements.
    Instructions:
    - Begin your response with "Manager Agent says:".
    - Identify the task and delegate it to the Coder or Runner as needed.
    - If you need user input, use the code word "TERMINATE" to pause the chat and ask for input.
    """,
    llm_config=llm_config,
    is_termination_msg=lambda msg: "TERMINATE" in msg["content"],  # Added termination message check
    human_input_mode="TERMINATE",  # Request human input when termination conditions are met
)

# Generic Coder Agent
coder_agent = ConversableAgent(
    name="coder_agent",
    system_message="""You are a helpful AI assistant.\n"
    "You use your coding skill to solve problems.\n"
    "You have access to a IPython kernel to execute Python code.\n"
    "You can suggest Python code in Markdown blocks, each block is a cell.\n"
    "The code blocks will be executed in the IPython kernel in the order you suggest them.\n"
    "All necessary libraries have already been installed. Provide runner agent with code.\n"
    """,
    llm_config=llm_config,
    is_termination_msg=lambda msg: "TERMINATE" in msg["content"],  # Added termination message check
    human_input_mode="TERMINATE",  # Request human input when termination conditions are met
)

# Generic Runner Agent
runner_agent = ConversableAgent(
    name="runner_agent",
    system_message="""You are the Runner Agent. Your task is to execute the Python code provided by the Coder.
    Instructions:
    - Begin your response with "Runner Agent says:".
    - Execute the provided code and return the results.
    """,
    llm_config=llm_config,
    is_termination_msg=lambda msg: "TERMINATE" in msg["content"],  # Added termination message check
    human_input_mode="TERMINATE",  # Request human input when termination conditions are met
    code_execution_config={"executor": NotebookExecutor()}
)

# Group Chat Configuration
def custom_speaker_selection_func(
    last_speaker: 'Agent', 
    groupchat: GroupChat
) -> Union[ConversableAgent, Literal['auto', 'manual', 'random', 'round_robin'], None]:
    """Define a customized speaker selection function.
    A recommended way is to define a transition for each speaker in the groupchat.

    Parameters:
        - last_speaker: Agent
            The last speaker in the group chat.
        - groupchat: GroupChat
            The GroupChat object
    Return:
        Return one of the following:
        1. an `Agent` class, it must be one of the agents in the group chat.
        2. a string from ['auto', 'manual', 'random', 'round_robin'] to select a default method to use.
        3. None, which indicates the chat should be terminated.
    """
    priority_order = [
        manager_agent,  
        coder_agent, 
        runner_agent
    ]
    
    if last_speaker in priority_order:
        next_index = (priority_order.index(last_speaker) + 1) % len(priority_order)
        return priority_order[next_index]
    return None

groupchat = GroupChat(
    agents=[manager_agent, coder_agent, runner_agent],  # Removed user agent from the group chat
    messages=[],
    max_round=50,
    speaker_selection_method=custom_speaker_selection_func
)

# Manager to handle the group chat
manager = GroupChatManager(groupchat=groupchat, llm_config=llm_config)

def main():
    logger.info("Starting the autogen app...")
    
    # Initiate the group chat
    manager.initiate_chat(
        recipient=manager_agent,  # Start the chat with the manager agent
        message="""
        Let's begin the workflow. The Manager will identify tasks, the Coder will write the necessary iPython code, and the Runner will execute that code.
        If at any point you need to ask the user for input, use the code word "TERMINATE" all caps to pause the chat.
        Manager, accept the overall description and terminate the chat with your first message asking user for the task.
    """
    )

if __name__ == "__main__":
    main()