import asyncio
import dspy
from dotenv import load_dotenv
from typing import List, Optional
from pydantic import BaseModel
from pydantic.fields import Field
from agent import Agent, preprocessAgent
from tool import WebsiteScrapper, InternetSearch, InternetAnswer
from utils import transform_schema_args_type
from action import Action
from tools_agents_selection import (
    GivenTaskAndContext,
    SelectedToolsAndAgents,
    ToolWithArgsValues,
    ToolResponse,
)

load_dotenv()
gpt_3 = dspy.OpenAI(model="gpt-4o-2024-05-13", max_tokens=4000, temperature=0)

dspy.configure(lm=gpt_3)


class InternetSearchBrowsedAnswers(BaseModel):
    """For a given task(along with some context), the agent did browse on the internet and find out the responses."""

    task: GivenTaskAndContext = Field(
        None, description="Task given to the agent to search on the web."
    )
    browsed_answers: str = Field(
        "",
        description="The browsed answers from the different websites after searching.",
    )


class InternetSearchAnswer(BaseModel):
    """The final search response after analyzing the task and browsed_answers"""

    answer: str = Field("", description="Final answer of the task.")


class FormulateInternetSearchAnswerSignature(dspy.Signature):
    """
    An agent was given a task for which the the agent uses the search tool for browsing internet to find out the answer.
    The agent did browse on the internet, went through different websites and collected the responses.

    Your job is to frame the right answer based on given task and analyzing browsed answers.

    Input:
    BrowsedAnswers: object containing task and browsed_answers
        task: the original task given to the agent.
        browsed_answers: agent browsed on the internet and collected the responses from different websites.

    Output:
    SearchAnswer: object containing answer.
        answer: The final framed nice answer for the given task after analyzing browsed_answers.

    """

    browsed_answers: InternetSearchBrowsedAnswers = dspy.InputField(
        desc="Browsed answer after searching on the internet."
    )
    search_answer: InternetSearchAnswer = dspy.OutputField(desc="Final answer.")


class SearchAgent(Agent):
    def __init__(
        self,
        name: Optional[str] = None,
        role: Optional[str] = None,
        tools: Optional[List] = None,
        team_agents: Optional[List] = None,
    ):
        default_name = "Internet Search Agent"
        default_role = """As a internet search agent for a given task, your role is to select tool and generate right search query to search on the web (by using tools provided) and generate the correct response."""
        default_tools = [InternetSearch(), InternetAnswer(), WebsiteScrapper()]
        default_team_agents = []

        Agent.__init__(
            self,
            name=name if name else default_name,
            role=role if role else default_role,
            tools=tools if tools else default_tools,
            team_agents=team_agents if team_agents else default_team_agents,
        )
        self.args = transform_schema_args_type(self.forward.__annotations__)
        self.trajectory = None
        self.pre_processesed_fields = preprocessAgent(agent=self)

        # signature & modules
        self._action = Action(preprocessed_fields=self.pre_processesed_fields)
        self._formulate_internet_search_answer = dspy.TypedChainOfThought(
            FormulateInternetSearchAnswerSignature
        )

    async def select_tools_and_agents(
        self, task: GivenTaskAndContext
    ) -> SelectedToolsAndAgents:
        return await self._action.select_right_tools_and_agents(task=task)

    async def scrape_url(self, search_result: dict) -> dict:
        url = search_result.get("url", "").strip()
        if url:
            # scraped_content = website_scrapper.run({"url": url})
            scraped_content = await self.pre_processesed_fields.tools_mapping[
                "website_scrapper"
            ]._arun(**{"url": url})
            if scraped_content is not None and len(scraped_content) > 0:
                scraped_content = scraped_content[0]
                page_content = scraped_content.page_content[:5000]
                if page_content.strip():
                    meta_data = scraped_content.metadata
                    search_result["scraped_content"] = page_content
                    search_result["metadata"] = meta_data

        return search_result

    async def scrape_urls(self, search_results: list):
        search_results = await asyncio.gather(
            *[self.scrape_url(search_result) for search_result in search_results]
        )
        return search_results

    async def formulate_search_answer(
        self, task: GivenTaskAndContext, browsed_answers: str
    ) -> str:
        response = self._formulate_internet_search_answer(
            browsed_answers=InternetSearchBrowsedAnswers(
                task=task, browsed_answers=browsed_answers
            )
        )
        return response.search_answer.answer

    async def run_internet_search(
        self, tool: ToolWithArgsValues, task: GivenTaskAndContext
    ) -> ToolResponse:
        # this is specifically built for running internet_search tool.

        search_results = await self.pre_processesed_fields.tools_mapping[
            tool.tool_name
        ]._arun(**tool.argument_values)

        if len(search_results) > 0:
            search_results = await self.scrape_urls(search_results=search_results)
            metadata = [x["metadata"] for x in search_results if x.get("metadata")]
            browsed_answers = "\n\n".join(
                [
                    f"Search Response_{idx}:{search_result['scraped_content'][:5000]}"
                    for idx, search_result in enumerate(search_results)
                    if search_result.get("scraped_content")
                ]
            )
            search_answer = await self.formulate_search_answer(
                task=task, browsed_answers=browsed_answers
            )
            response = {"search_answer": search_answer, "metadata": metadata}
        else:
            response = "No response from Internet Search Tool !!"

        return ToolResponse(tool=tool, response=response)

    async def forward(self, task: str, context: Optional[str] = ""):
        context = (
            self.pre_processesed_fields.background_story + context
            if context
            else self.pre_processesed_fields.background_story
        )

        selected_tools_and_agents_response = await self.select_tools_and_agents(
            task=GivenTaskAndContext(task=task, context=context)
        )

        tools_to_run, agents_to_execute = (
            selected_tools_and_agents_response.tools_to_run,
            selected_tools_and_agents_response.agents_to_execute,
        )

        tools_operation_response = await asyncio.gather(
            *[
                self.run_internet_search(
                    tool=tool, task=GivenTaskAndContext(task=task, context=context)
                )
                if tool.tool_name == "internet_search"
                else self._action.run_tool(tool)
                for tool in tools_to_run
            ]
        )
        agents_execution_response = await self._action.execute_agents(
            agents_to_execute=agents_to_execute
        )

        task_response = await self._action.generate_task_response(
            task=GivenTaskAndContext(task=task, context=context),
            tools_operation_response=tools_operation_response,
            agents_execution_response=agents_execution_response,
        )

        return task_response


if __name__ == "__main__":
    # from dspy.primitives.assertions import assert_transform_module, backtrack_handler
    # from functools import partial
    agent = SearchAgent()
    task = """

    Who won the t20 world cup 2024?
    """
    response = asyncio.run(agent(task=task))
    print(response)
