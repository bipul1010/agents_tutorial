import dspy
from pydantic import BaseModel, Field
from typing import List, Optional
from tools_agents_selection import (
    AvailableToolsAndAgents,
    ToolWithArgsTypes,
    AgentWithArgsTypes,
)
from functools import partial
from dspy.primitives.assertions import assert_transform_module, backtrack_handler


n_try = partial(backtrack_handler, max_backtracks=5)


class PreProcessedFields(BaseModel):
    background_story: str = Field("")
    tools_mapping: dict = Field({})
    agents_mapping: dict = Field({})
    tools_and_agents_args_type_formats: AvailableToolsAndAgents = Field(None)


class BaseAgent:
    def __init__(
        self, name: str, role: str, tools: Optional[List], team_agents: Optional[List]
    ):
        self.name = name
        self.role = role
        self.tools = tools
        self.team_agents = team_agents


class Agent(dspy.Module, BaseAgent):
    def __init__(
        self, name: str, role: str, tools: Optional[List], team_agents: Optional[List]
    ):
        dspy.Module.__init__(self)
        BaseAgent.__init__(
            self, name=name, role=role, tools=tools, team_agents=team_agents
        )


def create_background_story(agent: Agent):
    content = f"""You are a smart and expert agent with below properties:

    NAME: {agent.name}
    ROLE: {agent.role}
    
    
    """
    return content


def create_args_type(tools: List, team_agents: List) -> AvailableToolsAndAgents:
    available_tools = (
        [
            ToolWithArgsTypes(
                tool_name=tool.name,
                description=tool.description,
                argument_types=tool.args,
            )
            for tool in tools
        ]
        if len(tools) > 0
        else []
    )
    available_agents = (
        [
            AgentWithArgsTypes(
                agent_name=agent.name,
                role=agent.role,
                argument_types=agent.args,
            )
            for agent in team_agents
        ]
        if len(team_agents) > 0
        else []
    )
    return AvailableToolsAndAgents(
        available_tools=available_tools, available_agents=available_agents
    )


def preprocessAgent(agent: Agent) -> PreProcessedFields:
    # self.team_agents = [
    #     assert_transform_module(agent, n_try) for agent in self.team_agents
    # ]

    processed_fields = PreProcessedFields()

    processed_fields.background_story = create_background_story(agent)
    processed_fields.agents_mapping = {agent.name: agent for agent in agent.team_agents}
    processed_fields.tools_mapping = {tool.name: tool for tool in agent.tools}
    processed_fields.tools_and_agents_args_type_formats = create_args_type(
        tools=agent.tools, team_agents=agent.team_agents
    )
    return processed_fields


class Task(BaseModel):
    name: str = Field("Name of the task")
    description: str = Field("Steps and nuances to follow to finish this task")
    expected_output: str = Field(
        "An example which can help LLM model to get end result"
    )
