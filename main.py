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
import streamlit as st
import pandas as pd
import plotly.express as px
# from contextlib import asynccontextmanager
from dotenv import load_dotenv
from Dependancies import init_graph_state, create_image_dataset_graph, graph_runnable
from fastapi.responses import JSONResponse


print("The app is runing")

# FastAPI app creation with startup and shutdown events
#" @asynccontextmanager"
# async def lifespan(app: FastAPI):
#     # Load environment variables - in production use proper env management
#     global default_groq_api_key, default_openai_api_key

#     load_dotenv()

#     default_groq_api_key = os.getenv("GROQ_API_KEY", "")
#     default_openai_api_key = os.getenv("OPENAI_API_KEY", "")

#     if not default_groq_api_key or not default_openai_api_key:
#         print("Warning: API keys not found in environment variables. Users will need to provide them.")

#     # Initialize agents with default keys if available
#     if default_groq_api_key and default_openai_api_key:
#         app.state.agents = initialize_agents(default_groq_api_key, default_openai_api_key)
#         app.state.graph, app.state.router_chain = create_image_dataset_graph(app.state.agents)
#     else:
#         app.state.agents = None
#         app.state.graph = None
#         app.state.router_chain = None

#     yield

    # Cleanup code here if needed


# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Define models for API
class UserMessage(BaseModel):
    message: str
    session_id: str


class AIResponse(BaseModel):
    response: str
    has_images: bool = False
    image_urls: List[str] = []
    session_id: str


state = init_graph_state()
print("State initialized")


def process_message(user_input: str, current_state: Dict): # Get the string and the state
    """Process a user message and update the state"""   
    # Update state with new message
    current_state["messages"].append(HumanMessage(content=user_input))
    # Run the graph
    new_state = graph_runnable.invoke(current_state)
    return new_state



# API endpoints
@app.post("/chat", response_model=AIResponse)
async def chat(user_message: UserMessage):
    """Process a user message using the LangGraph workflow"""
    session_id = user_message.session_id # later update in database
    message = user_message.message

    # Get or create session
    # if session_id not in active_sessions:
    #     # Check if default agents are available
    #     if app.state.agents is None:
    #         raise HTTPException(
    #             status_code=400,
    #             detail="No API keys set. Please set API keys using the /set_api_keys endpoint first."
    #         )
    #     active_sessions[session_id] = create_new_session(session_id, app.state.agents, app.state.router_chain)

    # state = active_sessions[session_id]

    # Add user message to state
    state["messages"].append(HumanMessage(content=message))

    # Check for any feedback to sample images
    if state["current_agent"] == "sample_image_agent" and message:
        state["user_feedback"] = message

    # Process message through the graph- Invoke the GraphState
    new_state = process_message(message, state)

    # Extract response for the API
    ai_response = new_state["messages"][-1].content

    # Check if we have sample images to return
    has_images = False
    image_urls = []

    if "sample_images" in new_state and new_state["sample_images"]:
        has_images = True
        image_urls = new_state["sample_images"]

    # If we just completed dataset generation, include those images
    if new_state["current_agent"] == "dataset_generator_agent" and "generate_dataset" in str(new_state):
        has_images = True
        # Get the full list of generated images if available

    return AIResponse(
        response=ai_response,
        has_images=has_images,
        image_urls=image_urls,
        session_id=session_id
    )


class DatasetParameters(BaseModel):
    """Parameters for dataset generation"""
    num_images: int = Field(default=50, ge=5, le=200, description="Number of images to generate (5-200)")
    resolution: str = Field(default="256x256", description="Image resolution")
    session_id: str = Field(description="Session ID to apply these parameters to")


@app.post("/set_dataset_parameters")
async def set_dataset_parameters(params: DatasetParameters):
    """Set custom parameters for dataset generation"""
    session_id = params.session_id

    # # Check if session exists
    # if session_id not in active_sessions:
    #     raise HTTPException(status_code=404, detail="Session not found")

    # Update session state with the parameters provided by users
    state["num_images"] = params.num_images
    state["image_resolution"] = params.resolution

    # Inform user about the parameter change
    info_message = f"Dataset parameters updated: {params.num_images} images at {params.resolution} resolution."
    state["messages"].append(AIMessage(content=info_message))

    return {
        "status": "success",
        "message": f"Dataset parameters updated: {params.num_images} images at {params.resolution} resolution",
        "session_id": session_id
    }





class ApiKeys(BaseModel):
    """API keys for external services"""
    groq_api_key: str = Field(description="API key for Groq")
    openai_api_key: str = Field(description="API key for OpenAI")
    session_id: str = Field(description="Session ID to apply these keys to")


@app.post("/set_api_keys") #Function to set API keys from users in the state. and yeah Initialize agents
async def set_api_keys(keys: ApiKeys):
    """Set custom API keys for external services"""
    session_id = keys.session_id

    try:
        # agents = initialize_agents(keys.groq_api_key, keys.openai_api_key) # Initialize agents with the provided keys

        # Update API keys in the state
        state["groq_api_key"] = keys.groq_api_key
        state["openai_api_key"] = keys.openai_api_key

        return {
            "status": "success",
            "message": "API keys set and new session created",
            "session_id": session_id
        }
          
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to initialize agents with provided keys: {str(e)}"
        )


# @app.post("/set_api_keys")
# async def set_api_keys(keys: ApiKeys):
#     """Set custom API keys for external services"""
#     session_id = keys.session_id

#     # Check if session exists, create if not
#     if session_id not in active_sessions:
#         # Initialize agents with the provided keys
#         try:
#             agents = initialize_agents(keys.groq_api_key, keys.openai_api_key)
#             graph, router_chain = create_image_dataset_graph(agents)

#             # Store in app state for this session only
#             if not hasattr(app.state, 'custom_session_agents'):
#                 app.state.custom_session_agents = {}

#             app.state.custom_session_agents[session_id] = {
#                 'agents': agents,
#                 'graph': graph,
#                 'router_chain': router_chain
#             }

#             # Create new session with these agents
#             active_sessions[session_id] = create_new_session(
#                 session_id,
#                 agents,
#                 router_chain
#             )

#             # Update API keys in state
#             active_sessions[session_id]["groq_api_key"] = keys.groq_api_key
#             active_sessions[session_id]["openai_api_key"] = keys.openai_api_key

#             return {
#                 "status": "success",
#                 "message": "API keys set and new session created",
#                 "session_id": session_id
#             }
#         except Exception as e:
#             raise HTTPException(
#                 status_code=400,
#                 detail=f"Failed to initialize agents with provided keys: {str(e)}"
#             )
#     else:
#         # Update existing session with new keys
#         success = update_session_agents(session_id, keys.groq_api_key, keys.openai_api_key)

#         if success:
#             return {
#                 "status": "success",
#                 "message": "API keys updated for existing session",
#                 "session_id": session_id
#             }
#         else:
#             raise HTTPException(
#                 status_code=400,
#                 detail="Failed to update session with new API keys"
#             )




@app.get("/sessions/{session_id}")
async def get_session(session_id: str):
    """Get information about a specific session"""
    # if session_id not in active_sessions:
    #     raise HTTPException(status_code=404, detail="Session not found")

    # state = active_sessions[session_id]

    # Extract conversation messages for the frontend
    messages = []
    for msg in state["messages"]:
        if isinstance(msg, HumanMessage):
            messages.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            messages.append({"role": "assistant", "content": msg.content})

    return {
        "session_id": session_id,
        "messages": messages,
        "current_agent": state["current_agent"],
        "has_samples": len(state.get("sample_images", [])) > 0,
        "num_images": state["num_images"],
        "image_resolution": state["image_resolution"],
        "dataset_ready": state["dataset_ready"]
    }



# Will replace the session_id with the Database operations

# @app.post("/sessions/new")  
#     """Create a new session"""
#     import uuid

#     session_id = str(uuid.uuid4())
#     print(f"Creating new session with ID: {session_id}")

#     # Check if default agents are available
#     if app.state.agents is None:
#         print("Error: Agents not initialized. API keys are required.")
#         return {
#             "session_id": session_id,
#             "status": "keys_required",
#             "message": "Session created but API keys are required before use. Please call /set_api_keys endpoint."
#         }

#     active_sessions[session_id] = create_new_session(session_id, app.state.agents, app.state.router_chain)
#     print(f"Session {session_id} created successfully.")

#     return {
#         "session_id": session_id,
#         "status": "success",
#         "message": "New session created successfully"
#     }


# Also this one. Will Be replaec with Database operations

# @app.delete("/sessions/{session_id}")
# async def delete_session(session_id: str):
#     """Delete a session"""
#     if session_id not in active_sessions:
#         raise HTTPException(status_code=404, detail="Session not found")

#     # Remove the session
#     del active_sessions[session_id]

#     # Also clean up any custom agents if they exist
#     if hasattr(app.state, 'custom_session_agents') and session_id in app.state.custom_session_agents:
#         del app.state.custom_session_agents[session_id]

#     return {
#         "status": "success",
#         "message": f"Session {session_id} deleted successfully"
#     }



# @app.get("/health")
# async def health_check():
#     """Health check endpoint"""
#     return {
#         "status": "healthy",
#         "active_sessions": len(active_sessions),
#         "default_keys_available": app.state.agents is not None
#     }




# Load the dataset
@st.cache_data
def load_data():
    df = pd.read_csv('Energy_Consumption_India.csv')
    return df

df = load_data()

# Title and subtitle
st.title("Energy Consumption Dashboard")
st.subheader("Visualizing Energy Consumption in India")

# NEW: Add country selection
countries = df['Country'].unique()
selected_country = st.selectbox("Select a Country", countries)

# Filter based on selected country
df_country = df[df['Country'] == selected_country]

# NEW: Add year selection after filtering country
years = sorted(df_country['Year'].unique())
selected_year = st.selectbox("Select a Year", years)

# Filter based on selected year
df_filtered = df_country[df_country['Year'] == selected_year]

# Plotting
st.subheader(f"Energy Consumption by Sector in {selected_country} - {selected_year}")
fig = px.bar(df_filtered, x='Sector', y='Consumption', color='Sector',
             labels={'Consumption': 'Energy Consumption'},
             title='Energy Consumption by Sector')
st.plotly_chart(fig)


# Run the server if executed directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
