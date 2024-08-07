import dspy
from pydantic import BaseModel, Field
from typing import Optional, List
from dotenv import load_dotenv
from agent import Agent, preprocessAgent
from search_agent import SearchAgent
from langchain_core.prompts import PromptTemplate
from action import Action
from tools_agents_selection import GivenTaskAndContext

from dspy.primitives.assertions import assert_transform_module, backtrack_handler
from functools import partial
import asyncio
from trajectory import Trajectory, State
from utils import transform_schema_args_type


load_dotenv()
gpt_3 = dspy.OpenAI(model="gpt-4o-2024-05-13", max_tokens=4000, temperature=0)

dspy.configure(lm=gpt_3)


n_try = partial(backtrack_handler, max_backtracks=5)


class SlideOutline(BaseModel):
    """Outline about each slide"""

    title: str = Field("", description="Title of each slide.")
    content_outline: str = Field(
        "", description="Outline for creating the content of the slide."
    )


class SlideContent(BaseModel):
    """Final slide with right title and outline."""

    title: str = Field("", description="Title the slide.")
    content: str = Field("", description="Content on the slide.")


class PresentationOutlineOutput(BaseModel):
    """Outline for building the presentation. Make sure, the total slides doesn't exceed 15."""

    outline: List[SlideOutline] = Field(
        [],
        description="List of slides with each slide havin title and content outline.",
    )


class PresentationOutlineSignature(dspy.Signature):
    """You are a smart AI agent for building great presentation. For a given task and context, build the outline of the presentation."""

    presentation_input: GivenTaskAndContext = dspy.InputField(
        desc="task and context for building outline. "
    )
    presentation_outline_output: PresentationOutlineOutput = dspy.OutputField(
        desc="Outline of the presentation."
    )


class PresentationContent(BaseModel):
    """Final presentation including the list of slides containing title and content."""

    presentation: List[SlideContent] = Field(
        [], description="List of slides containing title and content."
    )


class ReviewPresentationSignature(dspy.Signature):
    """You are a smart presentation builder. You are given a presentation which containing multiple slides.
    Every slide has its title and content.

    Your job is to review the 'current_presentation' and give us the cleaned i.e. 'cleaned_presentation'. While reviewing please take care of the following points:

    1) Avoid and remove repetitive content on multiple slides. Every slide should be mostly having unique information.
    2) Every slide content should be crisp and contains only relevant information. Avoid unnecessary explanation.
    3) Remove all the reference links if present on multiple slides, and put this in a single slide containing Reference links.
    4) The title and content should be matching in the slide.
    5) Put the slide content on the bullet points and subpoints if you think it looks good in that way otherwise ignore.


    """

    current_presentation: PresentationContent = dspy.InputField(
        desc="The current presentation with multiple slides containing title and slide_content."
    )
    cleaned_presentation: PresentationContent = dspy.OutputField(
        desc=" A cleaned presentation after throughly reviewing multiple times. "
    )


class PresentationAIAgent(Agent):
    def __init__(
        self,
        name: Optional[str] = None,
        role: Optional[str] = None,
        tools: Optional[List] = None,
        team_agents: Optional[List] = None,
    ):
        default_name = "Presentation AI Agent"
        default_role = """You are an expert in building presentation slides. Based on the task given, you research thoroughly using tools and also coordinate with your team_agents whenever required. 
        Build the title of each slide and its content as well. Based on the plans, build each slide with relevant content.
        """
        default_tools = []
        default_team_agents = [SearchAgent()]

        Agent.__init__(
            self,
            name=name if name else default_name,
            role=role if role else default_role,
            tools=tools if tools else default_tools,
            team_agents=team_agents if team_agents else default_team_agents,
        )

        self.args = transform_schema_args_type(self.forward.__annotations__)
        self.trajectory = None
        self.pre_processesed_fields = preprocessAgent(self)

        # prompt_template
        self.slide_prompt = PromptTemplate.from_template(
            """Generate content of the slide (basically one-pager slide) of the overall presentation based on title and outline provided. If possible please also include the sources (including links) of the content.

            title: {title}
            outline: {outline}
            
            """
        )

        ##Signatures & Modules.
        self._outline_presentation = dspy.TypedChainOfThought(
            PresentationOutlineSignature
        )
        self._review_presentation = assert_transform_module(
            dspy.TypedChainOfThought(ReviewPresentationSignature, max_retries=10),
            partial(backtrack_handler, max_backtracks=5),
        )
        self._action = Action(preprocessed_fields=self.pre_processesed_fields)

    def build_presentation_outline(
        self, task: str, context: str
    ) -> PresentationOutlineOutput:
        response = self._outline_presentation(
            presentation_input=GivenTaskAndContext(task=task, context=context)
        )
        state = State(
            task=GivenTaskAndContext(task=task, context=context),
            response=response.presentation_outline_output,
        )
        self.trajectory.add_state(state)
        return response.presentation_outline_output

    async def generate_each_slide(self, slide_outline: SlideOutline) -> SlideContent:

        # print(f"Slide outline: {slide_outline}")
        task = self.slide_prompt.format(
            title=slide_outline.title, outline=slide_outline.content_outline
        )
        task_context = GivenTaskAndContext(
            task=task, context=self.pre_processesed_fields.background_story
        )
        response = await self._action(task=task_context)
        self.trajectory.add_state(self._action.state)
        # print(f"Slide content: {response}")
        return SlideContent(title=slide_outline.title, content=response)

    async def review_presentation(
        self, presentation: List[SlideContent]
    ) -> PresentationContent:
        response = self._review_presentation(
            current_presentation=PresentationContent(presentation=presentation)
        )

        dspy.Suggest(
            isinstance(response.cleaned_presentation, PresentationContent),
            "Cleaned Presentation should have type PresentationContent",
        )
        state = State(
            task=GivenTaskAndContext(
                task=PresentationContent(presentation=presentation)
            ),
            response=response.cleaned_presentation,
        )
        self.trajectory.add_state(state)
        return response.cleaned_presentation

    async def forward(self, task: str, context: Optional[str] = None):
        context = (
            self.pre_processesed_fields.background_story + context
            if context
            else self.pre_processesed_fields.background_story
        )

        self.trajectory = Trajectory(
            task=GivenTaskAndContext(task=task, context=context),
            resources=self.pre_processesed_fields.tools_and_agents_args_type_formats,
        )

        # print(context, self.pre_processesed_fields)

        presentation_outline = self.build_presentation_outline(
            task=task, context=context
        )

        presentation = await asyncio.gather(
            *[
                self.generate_each_slide(slide_outline)
                for slide_outline in presentation_outline.outline
            ]
        )

        # review presentation.
        presentation_after_review = await self.review_presentation(
            presentation=presentation
        )
        return presentation_after_review.presentation


if __name__ == "__main__":
    agent = PresentationAIAgent()
    # print(gpt_3.inspect_history(n=10))

    y = asyncio.run(
        agent.forward(
            task="AI agentic workflow for Healthcare. Its value addition, challenges and how to overcome with the help of network of agents.",
            context="Generate 3 pager slides.",
        )
    )
    print(y)
