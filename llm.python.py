# autogen_app.py
import logging
from autogen import ConversableAgent, GroupChat, GroupChatManager, Agent
from autogen.coding import CodeExecutor, CodeBlock, CodeResult, MarkdownCodeExtractor
from typing import List, Union, Literal

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
    def code_extractor(self) -> MarkdownCodeExtractor:
        # Extract code from markdown blocks.
        return MarkdownCodeExtractor()

    def __init__(self) -> None:
        # Do not assume IPython availability; initialize without it
        self._ipython = None

    def execute_code_blocks(self, code_blocks: List[CodeBlock]) -> CodeResult:
        log = ""
        exit_code = 0  # Initialize exit code
        for code_block in code_blocks:
            try:
                # Execute the code block and capture the output
                log += code_block.code + "\n"
                # Execute the code block using exec() and capture the output
                exec_globals = {}
                exec(code_block.code, exec_globals)
                log += f"Executed code block successfully.\n"
            except Exception as e:
                log += f"Error executing code block: {e}\n"
                exit_code = 1  # Set exit code to 1 if an error occurs
        return CodeResult(exit_code=exit_code, output=log)  

# AWS Data Retrieval Agent using Code Executor
aws_agent = ConversableAgent(
    name="aws_agent",
    system_message="""You are the AWS Data Retrieval Agent. Your task is to fetch data from AWS using Boto3.
    Instructions:
    - Begin your response with "AWS Agent says:".
    - Use provided AWS credentials to access the required data.
    - Share the retrieved data in the group chat.
    - Ensure to execute any necessary code to retrieve data, do not invent results.
    """,
    llm_config=llm_config,
    human_input_mode="NEVER",
    code_execution_config={"executor": NotebookExecutor()}
)

# GitHub Repository Retrieval Agent using Code Executor
github_agent = ConversableAgent(
    name="github_agent",
    system_message="""You are the GitHub Repository Retrieval Agent. Your task is to fetch Terraform and Terragrunt repositories using the GitHub API.
    Instructions:
    - Begin your response with "GitHub Agent says:".
    - Use provided GitHub credentials to access the repositories.
    - Share the retrieved repository data in the group chat.
    - Ensure to execute any necessary code to retrieve data, do not invent results.
    """,
    llm_config=llm_config,
    human_input_mode="NEVER",
    code_execution_config={"executor": NotebookExecutor()}
)

# Manager Agent
manager_agent = ConversableAgent(
    name="manager_agent",
    system_message="""You are the Manager Agent. Your task is to oversee the process and ensure all agents are performing their tasks correctly.
    Instructions:
    - Begin your response with "Manager Agent says:".
    - Monitor the progress of other agents and provide guidance as needed.
    """,
    llm_config=llm_config,
    human_input_mode="NEVER",
)

# Data Decision Agent
data_decision_agent = ConversableAgent(
    name="data_decision_agent",
    system_message="""You are the Data Decision Agent. Your task is to decide what data to move on to the next step.
    Instructions:
    - Begin your response with "Data Decision Agent says:".
    - Analyze the data provided by other agents and decide what to query next.
    """,
    llm_config=llm_config,
    human_input_mode="NEVER",
)

# Datadog Query Agent using Code Executor
datadog_agent = ConversableAgent(
    name="datadog_agent",
    system_message="""You are the Datadog Query Agent. Your task is to query Datadog for relevant metrics.
    Instructions:
    - Begin your response with "Datadog Agent says:".
    - Use provided Datadog credentials to perform the queries.
    - Ensure to execute any necessary code to retrieve data, do not invent results.
    """,
    llm_config=llm_config,
    human_input_mode="NEVER",
    code_execution_config={"executor": NotebookExecutor()}
)

# Code Committer Agent
committer_agent = ConversableAgent(
    name="committer_agent",
    system_message="""You are the Committer Agent. Your task is to adjust the code based on the decisions made and prepare it for submission.
    Instructions:
    - Begin your response with "Committer Agent says:".
    - Make necessary code adjustments and prepare for submission.
    """,
    llm_config=llm_config,
    human_input_mode="NEVER",
)

# Code Verifier Agent
verifier_agent = ConversableAgent(
    name="verifier_agent",
    system_message="""You are the Verifier Agent. Your task is to verify the code before submission.
    Instructions:
    - Begin your response with "Verifier Agent says:".
    - Ensure the code meets all quality standards before submission.
    """,
    llm_config=llm_config,
    human_input_mode="NEVER",
)

# GitHub Submission Agent using Code Executor
submission_agent = ConversableAgent(
    name="submission_agent",
    system_message="""You are the Submission Agent. Your task is to submit the code using the GitHub API.
    Instructions:
    - Begin your response with "Submission Agent says:".
    - Use provided GitHub credentials to submit the code.
    - Ensure to execute any necessary code to submit data, do not invent results.
    """,
    llm_config=llm_config,
    human_input_mode="NEVER",
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
        data_decision_agent, 
        aws_agent, 
        github_agent, 
        datadog_agent, 
        committer_agent, 
        verifier_agent, 
        submission_agent
    ]
    
    if last_speaker in priority_order:
        next_index = (priority_order.index(last_speaker) + 1) % len(priority_order)
        return priority_order[next_index]
    return None

groupchat = GroupChat(
    agents=[aws_agent, github_agent, manager_agent, data_decision_agent, datadog_agent, committer_agent, verifier_agent, submission_agent],
    messages=[],
    max_round=50,
    speaker_selection_method=custom_speaker_selection_func,
)

# Manager to handle the group chat
manager = GroupChatManager(groupchat=groupchat, llm_config=llm_config)

def main():
    logger.info("Starting the autogen app...")
    # Pass necessary credentials to agents
    aws_agent.credentials = {"aws_access_key_id": "123", "aws_secret_access_key": "321"}
    github_agent.credentials = {"github_token": "456"}
    datadog_agent.credentials = {"datadog_api_key": "789"}
    submission_agent.credentials = {"github_token": "1011"}
    
    # Initiate the group chat
    manager.initiate_chat(
        recipient=manager_agent,
        message="""
        Let's begin the data retrieval and processing workflow.
        aws_agent gets data from aws with boto library, he runs code to do so, so you need to tell him which code to run,
        github_agent gets terraform and terragrunt private repositories with github api, he runs code to do so, so you need to tell him which code to run,
        manager_agent is manager oversees the process 
        data_decision_agent decides what data to move onto (like query something on api)
        datadog_agent queries datadog for aws metrics, he runs code to do so, so you need to tell him which code to run,
        committer_agent is commiter adjusts the terraform code with suggestions 
        verifier_agent verifies the code
        submission_agent submits the code with github api to private repo,  he runs code to do so, so you need to tell him which code to run,
        all of you are in the group chat speak in order. Address with code pieces destined to particular agent.
        Requests make in python. But the target code it terraform.
    """
    )

if __name__ == "__main__":
    main()