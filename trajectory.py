from typing import List, Optional, Any
from pydantic import BaseModel
from pydantic.fields import Field
from tools_agents_selection import (
    GivenTaskAndContext,
    AgentResponse,
    ToolResponse,
    AvailableToolsAndAgents,
)


class State(BaseModel):

    task: GivenTaskAndContext = Field(None, description="Task of the state")
    tools_used: List[ToolResponse] = Field([], description="Tools used")
    agents_interaction: List[AgentResponse] = Field(
        [], description="Team agents collaborated ."
    )
    response: Any = Field("", description="State response.")
    reward: int = Field(-1, description="Reward of the state for executing the task.")


class Trajectory(BaseModel):
    task: GivenTaskAndContext = Field(None, description="Original task given.")
    resources: AvailableToolsAndAgents = Field(
        None, description="Available resources.  "
    )
    states: List[State] = Field([], description="List of states.")
    response: Any = Field(None, description="Final response")
    reward: int = Field(-1, description="Reward based on the final response. ")

    def add_state(self, state: State):
        self.states.append(state)

    def get_last_state(self):
        return self.states[-1] if self.states else None
