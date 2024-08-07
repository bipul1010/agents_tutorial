import dspy
from typing import List, Optional
from tools_agents_selection import (
    GivenTaskAndContext,
    SelectToolsAndAgentsSignature,
    SelectedToolsAndAgents,
    ToolWithArgsValues,
    AgentWithArgsValues,
    GenerateTaskResponseSignature,
    ToolResponse,
    AgentResponse,
)
from agent import PreProcessedFields
from utils import (
    create_content_for_agents_execution_response,
    create_content_for_tools_operation_response,
)
import asyncio
from trajectory import State


class Action(dspy.Module):
    def __init__(self, preprocessed_fields: PreProcessedFields):
        super().__init__()

        self.preproessed_fields = preprocessed_fields

        """signature"""
        self._select_tools_and_agents = dspy.TypedChainOfThought(
            SelectToolsAndAgentsSignature
        )
        self._generate_task_response = dspy.TypedChainOfThought(
            GenerateTaskResponseSignature
        )
        self.state = None

    async def run_tool(self, tool: ToolWithArgsValues) -> ToolResponse:
        tool_name = tool.tool_name
        tool_args = tool.argument_values

        if tool_name not in self.preproessed_fields.tools_mapping:
            raise KeyError(
                f"{tool_name} has to be present in {self.preproessed_fields.tools_mapping}"
            )
        response = await self.preproessed_fields.tools_mapping[tool_name]._arun(
            **tool_args
        )

        return ToolResponse(tool=tool, response=response)

    async def run_tools(
        self, tools_to_run: List[ToolWithArgsValues]
    ) -> List[ToolResponse]:
        # output_response = {tool.name: self.run_tool(tool) for tool in tools_to_run}
        # values = await asyncio.gather(*output_response.values())

        responses = await asyncio.gather(
            *[self.run_tool(tool) for tool in tools_to_run]
        )
        return responses

    async def execute_agent(self, agent: AgentWithArgsValues) -> AgentResponse:
        agent_name = agent.agent_name
        agent_args = agent.argument_values
        if agent_name not in self.preproessed_fields.agents_mapping:
            raise KeyError(
                f"{agent_name} has to be present in {self.preproessed_fields.agents_mapping}"
            )

        response = await self.preproessed_fields.agents_mapping[agent_name].forward(
            **agent_args
        )

        return AgentResponse(agent=agent, response=response)

    async def execute_agents(
        self, agents_to_execute: List[AgentWithArgsValues]
    ) -> List[AgentResponse]:
        # output_response = {
        #     agent.agent_name: self.execute_agent(agent) for agent in agents_to_execute
        # }
        # values = await asyncio.gather(*output_response.values())
        # return dict(zip(output_response.keys(), values))
        responses = await asyncio.gather(
            *[self.execute_agent(agent) for agent in agents_to_execute]
        )
        return responses

    async def select_right_tools_and_agents(
        self, task: GivenTaskAndContext
    ) -> SelectedToolsAndAgents:
        response = self._select_tools_and_agents(
            task_context=task,
            available_tools_and_agents=self.preproessed_fields.tools_and_agents_args_type_formats,
        )
        if hasattr(response, "selected_tools_and_agents"):
            return response.selected_tools_and_agents
        else:
            return ValueError("The response doesn't contain selected_tools_and_agents")

    async def generate_task_response(
        self,
        task: GivenTaskAndContext,
        tools_operation_response: List[ToolResponse],
        agents_execution_response: List[AgentResponse],
    ) -> str:

        response = self._generate_task_response(
            task=task,
            tools_operation_response=create_content_for_tools_operation_response(
                tools_operation_response
            ),
            agents_execution_response=create_content_for_agents_execution_response(
                agents_execution_response
            ),
        )

        return response.final_response

    async def forward(self, task: GivenTaskAndContext) -> str:

        self.state = State(task=task)  # initializing state

        selected_tools_and_agents_response = await self.select_right_tools_and_agents(
            task=task
        )

        tools_to_run, agents_to_execute = (
            selected_tools_and_agents_response.tools_to_run,
            selected_tools_and_agents_response.agents_to_execute,
        )
        # print(f"Selected tools :{tools_to_run} | Agents :{agents_to_execute}")

        tools_operation_response, agents_execution_response = await self.run_tools(
            tools_to_run
        ), await self.execute_agents(agents_to_execute)

        task_response = await self.generate_task_response(
            task=task,
            tools_operation_response=tools_operation_response,
            agents_execution_response=agents_execution_response,
        )

        # update state
        self.state.tools_used = tools_operation_response
        self.state.agents_interaction = agents_execution_response
        self.state.response = task_response  # update state

        return task_response
