import asyncio
from langchain_community.tools import tavily_search
from pydantic import BaseModel
from typing import Any, Dict
from pydantic.fields import Field
from utils import (
    custom_extractor,
    transform_schema_args_type,
)
from langchain_community.tools.tavily_search import TavilySearchResults, TavilyAnswer
from langchain_community.document_loaders.recursive_url_loader import RecursiveUrlLoader
from dotenv import load_dotenv


load_dotenv()


class LangChainCommunityTools:
    tavily_search = TavilySearchResults()
    tavily_answer = TavilyAnswer()


class WebsiteScrapper(BaseModel):
    name: str = Field("website_scrapper", description="Tool Name.")
    description: str = Field(
        "Scrape website of given url to extract content.The tool also go recursive to extract child links and extract content from there as well.",
        description="Description about the tool.",
    )
    args: Dict[str, Any] = Field(default_factory=dict)

    def __init__(self, /, **data: Any) -> None:
        super().__init__(**data)
        self.args = transform_schema_args_type(self._run.__annotations__.copy())

    def _run(self, url: str, max_depth: int = 2):
        loader = RecursiveUrlLoader(url=url, extractor=custom_extractor)
        result = loader.load()
        return result

    async def _arun(self, url: str, max_depth: int = 2):
        return self._run(url=url, max_depth=max_depth)


class InternetSearch(BaseModel):
    name: str = Field("internet_search")
    description: str = Field(LangChainCommunityTools.tavily_search.description)

    args: Dict[str, Any] = Field(default_factory=dict)

    def __init__(self, /, **data: Any) -> None:
        super().__init__(**data)
        self.args = transform_schema_args_type(self._run.__annotations__.copy())
        self.description = (
            self.description
            + """The tool will search and browse internet.Use this tool only when the query requires analysis in depth and understanding else use internet_answer for direct answer.
                To run this tool, generate the right query with less than 100 characters.
            """
        )

    def _run(self, query: str):
        return LangChainCommunityTools.tavily_search._run(query=query)

    async def _arun(self, query: str):
        return self._run(query=query)


class InternetAnswer(BaseModel):
    name: str = Field("internet_answer")
    description: str = Field(LangChainCommunityTools.tavily_answer.description)

    args: Dict[str, Any] = Field(default_factory=dict)

    def __init__(self, /, **data: Any) -> None:
        super().__init__(**data)
        self.args = transform_schema_args_type(self._run.__annotations__.copy())
        self.description = (
            self.description
            + """"It won't browse the web. Use this tool only when the query requires straightforward answer without any analysis else use internet_search for depth and understanding."
            """
        )

    def _run(self, query: str):
        return LangChainCommunityTools.tavily_answer._run(query=query)

    async def _arun(self, query: str):
        return self._run(query=query)


if __name__ == "__main__":
    tool = InternetAnswer()
    print(tool.args, tool.name, tool.description)
    # print(
    #     tool._run(
    #         **{"url": "https://dspy-docs.vercel.app/docs/quick-start/installation"}
    #     )
    # )
    print(asyncio.run(tool._arun(**{"query": "Where is Taj Mahal?"})))
