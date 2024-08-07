from typing import List, Dict, get_args, get_origin, _GenericAlias
from bs4 import BeautifulSoup
from langchain.tools import tool
from langchain_core.runnables.utils import Output
from tools_agents_selection import ToolResponse, AgentResponse
from langchain_community.document_loaders.recursive_url_loader import RecursiveUrlLoader


def custom_extractor(html_content):
    soup = BeautifulSoup(html_content, "html.parser")
    return soup.get_text()


def create_context_for_agent(name: str, role: str, tool: str):
    context = f"""You are an agent with below details.
    
    Name: {name} \n\n


    Role: {role} \n\n

    and you are also provided with below tools:\n\n

    Tools: 

    {tool}
    
    """

    return context


def create_content_for_tools_executed_response(tools_executed_response_dict: dict):
    content = """The following tools have generated following responses: \n\n
    
    
    """
    for idx, (tool_name, response) in enumerate(tools_executed_response_dict.items()):
        tool_content = f"""
        
        Tool[{idx}]_NAME: {tool_name} 
        Tool[{idx}]_RESPONSE: {response}
        
        
        """
        content = content + tool_content

    return content


def create_content_for_tools_operation_response(
    tools_operation_response: List[ToolResponse],
) -> str:

    if len(tools_operation_response) <= 0:
        return "No tools were used to generate response !"

    content = """The tools below have generated the corresonding responses after running:
    
    
    """

    for idx, (tool_response) in enumerate(tools_operation_response):
        tool_content = f"""
        
        Tool[{idx + 1}]_NAME: {tool_response.tool.tool_name} 
        Tool[{idx+1}]_RESPONSE: {tool_response.response}
        
        """
        content = content + tool_content
    return content


def create_content_for_agents_execution_response(
    agents_execution_responses: List[AgentResponse],
) -> str:

    if len(agents_execution_responses) <= 0:
        return "No agents were collaborated to generate response !"

    content = """The team agents below have generated the corresonding responses after execution:
    
    
    """

    for idx, (agent_response) in enumerate(agents_execution_responses):
        agent_content = f"""
        
        AGENT[{idx+1}]_NAME: {agent_response.agent.agent_name} 
        AGENT[{idx+1}]_RESPONSE: {agent_response.response}
        
        """
        content = content + agent_content
    return content


def create_content_for_validate_tool_response(tool_name, tool_response):
    content = f"""You have used below tool with tool_name and have generated response after running the tool.
    
    TOOL_NAME: {tool_name}
    TOOL_RESPONSE: {tool_response}
    
    """

    return content


def python_type_to_json_type(py_type: type) -> str:
    type_mapping = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        list: "array",
        dict: "object",
    }
    return type_mapping.get(
        py_type, "string"
    )  # Default to string if type is not recognized


def transform_schema_args_type(args):
    output_schema = dict()
    for key, value in args.items():
        field_schema = {"title": key.capitalize()}
        # print(key, value)
        if isinstance(value, type):
            field_schema["type"] = python_type_to_json_type(value)
        elif isinstance(value, _GenericAlias):
            origin = get_origin(value)
            args = get_args(value)
            # print(origin, args, value.__name__)
            if value.__name__.lower() == "optional":
                field_schema["type"] = python_type_to_json_type(args[0])
                field_schema["optional"] = True
            else:
                field_schema["type"] = python_type_to_json_type(origin)
                if args is not None and len(args) > 0:
                    field_schema["items"] = {"type": python_type_to_json_type(args[0])}

        output_schema[key] = field_schema

    return output_schema


if __name__ == "__main__":
    print("hello")
    # x = scrape_website_text_content.run(
    #     {
    #         "url": "https://www.analyticsvidhya.com/blog/2024/05/agentic-ai-demystified-the-ultimate-guide-to-autonomous-agents/"
    #     }
    # )
    # print(x[0].page_content)
