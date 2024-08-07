from dataclasses import field
from typing import Any, List, Optional
import dspy
from pydantic import BaseModel
from pydantic.fields import Field


class GivenTaskAndContext(BaseModel):
    """
    You are provided with the task and context

    """

    task: Any = Field("", description="task to execute.")
    context: str = Field(
        "", description="the context which can help AI in executing task better."
    )


class ToolWithArgsTypes(BaseModel):
    """Tool with below properties:

    1) tool_name: Name of the tool
    2) description: The description about the tool, what it does and the reason behind running this tool.

    3) argument_types: The type of argument it requires to run this tool. It will be a dictionary, where dict key and dict value represent the argument name and its type respectively.


    For example:


    -> tool_name: 'get_weather'
    -> description: 'Get the description of weather using get_weather tool.'
    -> argument_types:{
                "latitude": {"title": "Latitude", "type": "float"},
                "longitutude": {"title": "Longitutude", "type": "float"},
            }
    Here, 'latitude' and 'longitutude' are the arguments required and both their types are float.

    """

    tool_name: str = Field("", description="Name of the tool")
    description: str = Field("", description="Description about the tool")
    argument_types: dict = Field(
        {}, description="Arguments and its type required to run tool"
    )


class ToolWithArgsValues(BaseModel):
    """Tool with argument values to run.

        1) tool_name: Name of the tool
        2) argument_values: The value of argument which will be provided to the tool to run. It will be a dictionary, where dict key and dict value are argument name and argument value respectively.

    For example:
    -> tool_name: 'get_weather'
    -> argument_values: {"latitude":34.5,"longitutude":46.3}
    """

    tool_name: str = Field("", description="Name of the tool")
    argument_values: dict = Field(
        {}, description="Arguments and its value to run the tool"
    )


class ToolResponse(BaseModel):
    tool: ToolWithArgsValues = Field(
        None, description="Tool ran using its argument values."
    )
    response: Any = Field(None, description="Response from the tool.")


class AgentWithArgsTypes(BaseModel):
    """Agent with below properties

    1) agent_name: Name of the agent.
    2) role: Role of the agent to execute task.
    3) argument_types: Type of argument it requires to execute agent. It will be a dictionary, where dict key and dict value represent the argument name and its type.

    For example:
    -> agent_name: 'Search Agent'
    -> role: 'You are a smart search agent which browse internet for a task and provide relevant answer.'
    -> argument_types: {'task': {'title': 'Task', 'type': 'string'}, 'context': {'title': 'Context', 'type': 'string', 'optional': True}}

    Here, 'task' and 'context' are the arguments required. The type of 'task' is string and type of 'context' is string which is optional.

    """

    agent_name: str = Field("", description="Name of the agent")
    role: str = Field("", description="Role about the agent.")
    argument_types: dict = Field(
        {}, description="Arguments and its types required to execute agent"
    )


class AgentWithArgsValues(BaseModel):
    """Agent with argument values to execute.

    1) agent_name: Name of the agent.
    2) argument_values: The value of argument which will be provided to the agent to execute. It will be a dictionary, where dict key and dict value are argument name and argument value respectively.

    For example:

    -> agent_name: 'Search Agent'
    -> argument_values: {'task': 'Agentic workflow for Enterprises','context':'Include summary and challenges as well.'}
    """

    agent_name: str = Field("", description="Name of the agent")
    argument_values: dict = Field(
        {}, description="Arguments and its value to execute the agent"
    )


class AgentResponse(BaseModel):
    agent: AgentWithArgsValues = Field(
        None, description="Agent executed using its argument value."
    )
    response: Any = Field(None, description="Response from the agent.")


class AvailableToolsAndAgents(BaseModel):
    """Available tools and agents. The tools and agents are provided with the name and its argument types.

    1) available_tools: List of ToolWithArgsTypes where each ToolWithArgsTypes contains tool_name,description and argument_types.
    2) available_agents: List of AgentWithArgsTypes where each AgentWithArgsTypes contains agent_name,role and argument_types.
    """

    available_tools: List[ToolWithArgsTypes] = Field(
        [],
        description="List of ToolWithArgsTypes where each ToolWithArgsTypes represent tool_name,description and argument_types.",
    )
    available_agents: List[AgentWithArgsTypes] = Field(
        [],
        description="List of AgentWithArgsTypes where each AgentWithArgsTypes represent agent_name,description and argument_types.",
    )


class SelectedToolsAndAgents(BaseModel):
    """Select the right tools and agents from available_tools/available_agents based on the task and context provided.The objective is to run the tools or executing team agents which can help in executing task.
    Also, provide the brief reasoning behind selecting tools or agents.

    1) tools_to_run: List of ToolWithArgsValues where each ToolWithArgsValues contains tool_name and argument_values
    2) agents_to_execute: List of AgentWithArgsValues where each AgentWithArgsValues contains agent_name and argument_values.
    3) reasoning: A brief reason behind selecting the right tools or agents.

    Note: You can return tools_to_run or agents_to_execute as empty list if right available_tools or available_agents are not present for executing the task.

    """

    tools_to_run: List[ToolWithArgsValues] = Field(
        [],
        description="List of ToolWithArgsValues where each ToolWithArgsValues contains tool_name and argument_values",
    )
    agents_to_execute: List[AgentWithArgsValues] = Field(
        [],
        description="List of AgentWithArgsValues where each AgentWithArgsValues contains agent_name and argument_values.",
    )
    reasoning: str = Field(
        "", description="A brief reason behind selecting the right tools or agents."
    )


class SelectToolsAndAgentsSignature(dspy.Signature):
    """You are given a task along with some context. For execting the task, you are also supported with some available tools and also have team agents.

    Based on the tasks and context provided, your job is to select the right tools and agents with its argument values for executing the task.

    """

    task_context: GivenTaskAndContext = dspy.InputField(
        desc="Task and Context provided."
    )
    available_tools_and_agents: AvailableToolsAndAgents = dspy.InputField(
        desc="Available tools and your team agents."
    )
    selected_tools_and_agents: SelectedToolsAndAgents = dspy.OutputField(
        desc="Selected tools and agents for executing the task."
    )


class GenerateTaskResponseSignature(dspy.Signature):
    """You have to analyze the below responses and generate the right response for the given task.
    The below 'task' was given to the agent for which agent used following 'tools' and collaborate with its 'team agents'.

    'task': the original task given to the agent.
    'tools_operation_response': The response from each tool after running.
    'agents_execution_response': The response from each team agent after execution.

    'final_response': The final response after analyzing tools_operation_response and agents_execution_response for the given task. In the response, if possible also highlight the relevant links/sources for generating the response.

    """

    task: GivenTaskAndContext = dspy.InputField(desc="Task and context provided.")
    tools_operation_response: Optional[str] = dspy.InputField(
        desc="Responses after running tools.  "
    )
    agents_execution_response: Optional[str] = dspy.InputField(
        desc="Responses after executing agents."
    )
    final_response: str = dspy.OutputField(desc="Final response")


# from dotenv import load_dotenv

# load_dotenv()
# gpt_3 = dspy.OpenAI(model="gpt-4o-2024-05-13", max_tokens=4000, temperature=0)

# dspy.configure(lm=gpt_3)

# k = dspy.TypedChainOfThought(GenerateTaskResponseSignature)
# task = GivenTaskAndContext(
#     task="Generate a slide on AI agentic workflow",
#     context="Include summary and challenges",
# )

# tools_and_operation_response = """The tools below have generated the corresonding responses after running:

#     Tool[1]_NAME: "web_scrapper"
#     Tool[1]_RESPONSE: "AI agentic workflows integrate various facets of AI, such as machine learning, natural language processing, and predictive analytics, to automate and streamline complex tasks"

#     """


# response = k(
#     task=task,
#     tools_and_operation_response=tools_and_operation_response,
#     agents_execution_response="NO agents were used to get the response.",
# )
# print(response)
