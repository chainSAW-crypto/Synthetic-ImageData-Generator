from typing import TypedDict, Dict, List, Any, Optional, Literal
from fastapi import FastAPI, HTTPException, Body, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_groq import ChatGroq
from langchain_openai import OpenAI
import openai
from langgraph.graph import StateGraph, END
import re
import os
import json
# from contextlib import asynccontextmanager
from dotenv import load_dotenv


# Define the GraphState
class GraphState(TypedDict):
    messages: List[Any]  # The messages passed between user and assistant
    conversation_history: Dict  # A record of all conversation threads
    current_agent: str  # Track which agent is currently active
    prompt_list: List[str]  # Store generated prompts
    sample_images: List[str]  # Store URLs of sample images
    user_feedback: str  # Store user feedback on sample images
    dataset_ready: bool  # Flag to indicate if user wants the full dataset
    llm_model: str  # Track which LLM model is being used
    session_id: str  # Unique identifier for the conversation session
    num_images: int  # Number of images for the dataset
    image_resolution: str  # Resolution for generated images
    groq_api_key: str  # API key for Groq
    openai_api_key: str  # API key for OpenAI



# Global variables to store active sessions
active_sessions = {}


### Default API keys - will be overridden by user-provided keys ###

default_groq_api_key = os.getenv("groq_api_key")
default_openai_api_key = os.getenv("OPENAI_API_KEY")

# from google.colab import userdata

# default_groq_api_key = userdata.get("groq_api_key")
# default_openai_api_key = userdata.get("OPENAI_API_KEY")


if not default_groq_api_key or not default_openai_api_key:
    print("Error: Missing API keys. Ensure GROQ_API_KEY and OPENAI_API_KEY are set in the environment.")
    print(f"GROQ_API_KEY: {default_groq_api_key}")
    print(f"OPENAI_API_KEY: {default_openai_api_key}")


# Agent initialization
def initialize_agents(groq_api_key, openai_api_key):
    """Initialize all the agent components"""
    try:
        conversation_agent = ChatGroq(groq_api_key=groq_api_key, model="Gemma2-9b-It")
        prompt_generator = ChatGroq(groq_api_key=groq_api_key, model="Gemma2-9b-It")
        image_client = openai.OpenAI(api_key=openai_api_key)
        router_agent = ChatGroq(groq_api_key=groq_api_key, model="Gemma2-9b-It")
        return conversation_agent, prompt_generator, image_client, router_agent
    except Exception as e:
        print(f"Error initializing agents: {e}")
        raise



# Helper functions
def extract_scenario(state):
    """Extract the user's scenario from the conversation history"""
    relevant_messages = []

    for message in state["messages"]:
        if isinstance(message, HumanMessage):
            relevant_messages.append(message.content)

    # Concatenate all user messages into a single scenario description
    if relevant_messages:
        combined_scenario = " ".join(relevant_messages)
        # If too long, use only the most substantial messages
        if len(combined_scenario) > 500:
            # Find the longest message, which likely contains the main scenario
            longest_message = max(relevant_messages, key=len)
            return longest_message
        return combined_scenario

    return "generic dataset of various objects and scenes"


def get_current_user_question(state):
    """Extract the most recent user question from the state"""
    for message in reversed(state["messages"]):
        if isinstance(message, HumanMessage):
            return message.content
    return ""

# Generate a summary of conversation history
def summarize_conversation(state):
    """Create a brief summary of the conversation history for context"""
    history = []

    # Extract last few turns of conversation
    messages = state["messages"][-10:]  # Limit to last 10 messages

    for msg in messages:
        if isinstance(msg, HumanMessage):
            history.append(f"User: {msg.content}")
        elif isinstance(msg, AIMessage):
            # Truncate very long assistant responses
            content = msg.content
            if len(content) > 200:
                content = content[:197] + "..."
            history.append(f"Assistant: {content}")

    return "\n".join(history)

def parse_image_prompts(response_text):
    """Extract a list of prompts from the LLM response"""
    try:
        # Try to find a Python list in the text
        match = re.search(r'\[(.*?)\]', response_text, re.DOTALL)
        if match:
            prompts_text = match.group(1)
            # Safely evaluate the list
            return eval(f"[{prompts_text}]")

        # If no list is found, try to extract numbered items
        lines = response_text.split('\n')
        prompts = []
        for line in lines:
            # Look for numbered lines like "1. prompt" or "1) prompt"
            match = re.match(r'^\s*\d+[\.\)]\s*(.*)', line)
            if match:
                prompts.append(match.group(1).strip())

        if prompts:
            return prompts

        # If all else fails, split by newlines and filter
        return [line.strip() for line in lines if line.strip() and len(line.strip()) > 10]

    except Exception as e:
        print(f"Error parsing prompts: {e}")
        return []
        

      


# Init GraphState
def init_graph_state():
    """Create a new conversation session"""

    # Initialize the state with default values
    return GraphState(
        messages=[AIMessage(content="Hello! I'm your image dataset creation assistant. What kind of image dataset would you like to create today?")],
        conversation_history={},
        current_agent="conversation_agent",
        prompt_list=[],
        sample_images=[],
        user_feedback="",
        dataset_ready=False,
        llm_model="Gemma2-9b-It",
        conversation_agent=conversation_agent,
        prompt_generator=prompt_generator,
        image_client=image_client,
        router_agent=router_agent,
        router_chain=router_chain,
        session_id=session_id,
        num_images=50,  # Default number of images
        image_resolution="256x256",  # Default resolution
        groq_api_key=default_groq_api_key,
        openai_api_key=default_openai_api_key
    )



# Function to update agent components with new API keys
# def update_session_agents(session_id, groq_api_key, openai_api_key):
#     """Update the agents for a session with new API keys"""
#     if session_id not in active_sessions:
#         return False

#     state = active_sessions[session_id]

#     # Update API keys in state
#     state["groq_api_key"] = groq_api_key
#     state["openai_api_key"] = openai_api_key

#     # Re-initialize agents with new keys
#     try:
#         conversation_agent = ChatGroq(groq_api_key=groq_api_key, model="Gemma2-9b-It")
#         prompt_generator = ChatGroq(groq_api_key=groq_api_key, model="Gemma2-9b-It")
#         image_client = openai.OpenAI(api_key=openai_api_key)
#         router_agent = ChatGroq(groq_api_key=groq_api_key, model="Gemma2-9b-It")

#         # Update agents in state
#         state["conversation_agent"] = conversation_agent
#         state["prompt_generator"] = prompt_generator
#         state["image_client"] = image_client
#         state["router_agent"] = router_agent

#         # Also recreate the router chain
#         state["router_chain"] = create_router(router_agent)

#         active_sessions[session_id] = state
#         return True
#     except Exception as e:
#         print(f"Error updating session agents: {e}")
#         return False


# Agent handlers
def conversation_handler(state: GraphState):
    """The primary conversation agent to interact with the user"""

    user_message = state["messages"][-1].content

    system_message = """You are an AI assistant specializing in helping users create image datasets.
    Your role is to understand what kind of image dataset the user wants to create.

    Be conversational and helpful, focusing on understanding the user's requirements clearly.
    If the user seems to be describing a specific type of images, suggest they might want to see samples.
    If they've seen samples and like them, you can suggest generating the full dataset.

    Keep your responses concise and focused on helping create their image dataset.
    """

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_message),
        ("human", "{user_input}")
    ])

    conversation_agent = ChatGroq(groq_api_key=state["groq_api_key"], model="Gemma2-9b-It")
    chain = prompt | conversation_agent
    response = chain.invoke({"user_input": user_message})

    # Add to conversation history
    conversation_id = len(state["conversation_history"]) + 1
    state["conversation_history"][conversation_id] = {
        "question": user_message,
        "response": response
    }

    state["messages"].append(AIMessage(content=response.content))
    state["current_agent"] = "conversation_agent"

    return "router"




def sample_image_handler(state: GraphState):
    """Generate sample images for user review"""

    # Extract user scenario from conversation
    user_scenario = extract_scenario(state)

    # Generate a few sample prompts
    system = """You are a prompt generator for text-to-image models.
    Generate 3 diverse and detailed prompts for sample images based on the user scenario.
    Each prompt should be descriptive and optimized for image generation.
    Format your output as a Python list of exactly 3 strings: ["prompt1", "prompt2", "prompt3"]
    """

    prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", f"Generate 3 sample image prompts for scenario: {user_scenario}")
    ])

    prompt_generator = state["prompt_generator"]
    chain = prompt | prompt_generator
    response = chain.invoke({})

    # Extract the prompt list
    sample_prompts = parse_image_prompts(response.content)

    # Ensure we have exactly 3 prompts
    if len(sample_prompts) < 3:
        # Fill in missing prompts
        while len(sample_prompts) < 3:
            sample_prompts.append(f"{user_scenario} variation {len(sample_prompts)+1}")
    elif len(sample_prompts) > 3:
        # Trim to just 3 prompts
        sample_prompts = sample_prompts[:3]

    # Generate sample images
    sample_image_urls = []
    for prompt in sample_prompts:
        try:
            response = state["image_client"].images.generate(
                model="dall-e-2",
                prompt=prompt,
                size=state["image_resolution"],  # Use the resolution from state
                quality="standard",
                n=1
            )
            sample_image_urls.append(response.data[0].url)
        except Exception as e:
            print(f"Error generating image: {str(e)}")
            sample_image_urls.append(None)

    # Filter out any None values
    sample_image_urls = [url for url in sample_image_urls if url is not None]

    state["sample_images"] = sample_image_urls
    state["prompt_list"] = sample_prompts

    # Create response message with sample images
    response_text = "Here are some sample images based on your requirements:\n\n"
    for i, (prompt, url) in enumerate(zip(sample_prompts, sample_image_urls)):
        if url:
            response_text += f"Sample {i+1}:\nPrompt: {prompt}\n\n"

    response_text += f"How do these look? Would you like to make any adjustments before generating the full dataset of {state['num_images']} images?"

    state["messages"].append(AIMessage(content=response_text))
    state["current_agent"] = "sample_image_agent"

    return "router"



def prompt_generator_handler(state: GraphState):
    """Generate the full set of prompts based on user requirements"""

    # Extract user scenario from conversation
    user_scenario = extract_scenario(state)
    user_feedback = state.get("user_feedback", "")

    # Include user feedback if available
    scenario_with_feedback = user_scenario
    if user_feedback:
        scenario_with_feedback = f"{user_scenario}. User feedback on samples: {user_feedback}"

    # Use the custom number of images from state
    num_images = state["num_images"]

    system = f"""You are a prompt generator for text to image models. Your task is to generate {num_images} diverse prompts
    for text-to-image models to create a comprehensive dataset given the scenario.
    Each prompt should be descriptive and optimized for image generation.
    Make sure to cover different angles, perspectives, environments, and contexts.
    Format your output as a Python list of strings: ["prompt1", "prompt2", ...]
    """

    prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", f"Generate {num_images} diverse image prompts for scenario: {scenario_with_feedback}")
    ])

    prompt_generator = state["prompt_generator"]
    chain = prompt | prompt_generator
    response = chain.invoke({})

    # Extract the prompt list
    prompt_list = parse_image_prompts(response.content)

    # Ensure we have at least the required number of prompts
    if len(prompt_list) < num_images:
        # Fill in missing prompts
        while len(prompt_list) < num_images:
            prompt_list.append(f"{user_scenario} variation {len(prompt_list)+1}")

    state["prompt_list"] = prompt_list[:num_images]  # Limit to exactly the required number
    state["dataset_ready"] = True

    # Prepare response message
    response_text = f"I've generated {num_images} diverse prompts for your dataset based on your requirements.\n\n"
    response_text += "Here are a few examples:\n"
    for i, prompt in enumerate(prompt_list[:5]):
        response_text += f"{i+1}. {prompt}\n"

    response_text += f"\n...and {len(prompt_list)-5} more prompts.\n\n"
    response_text += f"Would you like me to proceed with generating all {num_images} images for your dataset now?"

    state["messages"].append(AIMessage(content=response_text))
    state["current_agent"] = "dataset_generator_agent"

    return "router"



def dataset_generator_handler(state: GraphState):
    """Generate the full dataset of images from the prompts"""

    try:
        num_images = state["num_images"]
        prompt_list = state["prompt_list"][:num_images]  # Ensure we use max the required number of prompts
        image_urls = []

        # Use custom resolution from state
        resolution = state["image_resolution"]

        # Create a progress update message
        progress_message = f"Starting to generate your dataset with {num_images} images at {resolution} resolution. This might take some time...\n"
        state["messages"].append(AIMessage(content=progress_message))

        # Generate images for each prompt
        for idx, prompt in enumerate(prompt_list):
            try:
                print(f"[{idx+1}/{len(prompt_list)}] Generating image for prompt: {prompt}")

                response = state["image_client"].images.generate(
                    model="dall-e-2",
                    prompt=prompt,
                    size=resolution,
                    quality="standard",
                    n=1
                )

                image_url = response.data[0].url
                image_urls.append(image_url)

            except Exception as e:
                print(f"Error generating image {idx+1}: {str(e)}")
                image_urls.append(None)

        # Filter out None values
        image_urls = [url for url in image_urls if url is not None]

        # Final response with dataset information
        final_message = f"✅ Dataset generation complete! Generated {len(image_urls)}/{num_images} images at {resolution} resolution.\n\n"

        # Include some sample URLs
        if image_urls:
            final_message += "Here are a few sample images from your dataset:\n"
            for i, url in enumerate(image_urls[:5]):
                if url:
                    final_message += f"Image {i+1}: {url}\n"

        final_message += "\nYou can download these images to assemble your complete dataset. Would you like me to help with anything else?"

        state["messages"].append(AIMessage(content=final_message))
        state["current_agent"] = "dataset_generator_agent"

        return "router"
    except Exception as e:
        print(f"Error in dataset_generator_handler: {e}")
        raise


# Defining the Router Properly
class RouterOutput(BaseModel):
    """Output schema for the router agent"""
    agent: Literal["conversation_agent", "sample_image_agent", "prompt_generator_agent", "dataset_generator_agent"]
    confidence: float = Field(description="Confidence level from 0.0 to 1.0")
    reasoning: str = Field(description="Brief explanation of why this agent was chosen")


llm = ChatGroq(groq_api_key=default_groq_api_key, model="Gemma2-9b-It")
router_agent = llm.with_structured_output(RouterOutput)

system = """You are an expert routing agent for an image dataset creation system.
Your job is to analyze the user's message and determine which of the following agents should handle it:

1. conversation_agent: For general conversation, questions about the system, or when the user is still describing their needs.

2. sample_image_agent: When the user is explicitly or implicitly requesting sample images, wants to see examples,
    or is describing specific image requirements that would benefit from visual confirmation before proceeding.

3. prompt_generator_agent: Generate the full set of prompts based on user requirements. If user aggres with the prompts images will be generated.

4. dataset_generator_agent: When the user is ready to generate the full dataset of images, expressing satisfaction with samples,
    or directly asking to proceed with dataset creation.

Analyze the content, context, and intent of the user's message, then select the most appropriate agent.

Output your decision as a JSON object with the following fields:
- agent: The name of the agent (one of the three above)
- confidence: Your confidence level in this decision (0.0 to 1.0)
- reasoning: A brief explanation of why you chose this agent
"""

router_prompt = ChatPromptTemplate.from_messages([
    ("system", system),
    ("human", "User message: {question}\n\n Conversation history summary: {history_summary}")
])

router_chain = router_prompt | router_agent



# Routing function
def route_question(state: GraphState):
    """Route the user question to the appropriate agent"""

    question = get_current_user_question(state)
    history_summary = summarize_conversation(state)

    # Make routing decision
    try:
        route_decision = router_chain.invoke({
            "question": question,
            "history_summary": history_summary
        })

        print(f"Router Decision: {route_decision}")

        # Update state to track which agent was selected
        state["current_agent"] = route_decision.agent

        # Direct routing based on router_chain output
        if route_decision.agent == "conversation_agent":
            print("---ROUTE QUESTION TO Conversation Agent---")
            return "conversation"
        elif route_decision.agent == "sample_image_agent":
            print("---ROUTE QUESTION TO Sample Image Agent---")
            return "generate_sample_images"  
        elif route_decision.agent == "dataset_generator_agent":
            print("---ROUTE QUESTION TO Dataset Generator Agent---")
            state["dataset_ready"] = True  # Set the flag if needed
            return "generate_prompts"  
        elif route_decision.agent == "dataset_generator_agent":
            print("---ROUTE QUESTION TO Dataset Generator Agent---")
            return "generate_dataset"

        # Map agent names to graph nodes
        # agent_to_node = {
        #     "conversation_agent": "conversation",
        #     "sample_image_agent": "generate_sample_images",
        #     "dataset_generator_agent": "generate_prompts"
        # }

        # return agent_to_node[route_decision.agent]

    except Exception as e:
        print(f"Routing error: {e}")
        # Default to conversation as fallback
        return "conversation"
    


# Create the graph
workflow = StateGraph(GraphState)

from langgraph.graph import END, StateGraph, START

# Add nodes
workflow.add_node("conversation", conversation_handler)
workflow.add_node("generate_sample_images", sample_image_handler)
workflow.add_node("generate_prompts", prompt_generator_handler)
workflow.add_node("generate_dataset", dataset_generator_handler)

# Adding conditional edges from router to specific agents
workflow.add_conditional_edges(
    START,
    route_question,
    {
        "conversation": "conversation",
        "generate_sample_images":"generate_sample_images",
        "generate_prompts": "generate_prompts",
        "generate_dataset": "generate_dataset"
    },
)

# workflow.add_edge("router", "conversation")
# workflow.add_edge("router", "generate_sample_images")
# workflow.add_edge("router", "generate_prompts")


workflow.add_edge( "conversation", END)
workflow.add_edge( "generate_sample_images", END)
workflow.add_edge( "generate_prompts", END)
workflow.add_edge( "generate_dataset", END)

graph_runnable = workflow.compile()