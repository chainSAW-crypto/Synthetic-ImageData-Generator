# Gen AI based Data Generator for Researchers and ML engineers 

To solve the Gap of tedious manual image generation to train on single scenario, we provide the AI powered application to automate the whole process and make it a one stop solution for all creation, modification and augmentation needs for a image Data. 
We are keeping it open source so that the Community can utilize its capabilities, in order to Solve Dataset Needs for their ML applications.
The Project Utilizes the Gen-AI capabilities to Generate Data to match real world Scenarios. 
Automating the tedious task of collecting Data and scraping. Agentic Scraping of Data, reflecting explainability over the provided results. eg. Results like Perplexity. 
Integration of Generative capabilities and Augmentation to create versatile databases to cater your research needs.

A FastAPI-based service that generates synthetic image datasets using LangGraph workflow and AI models. This project combines the power of language models and image generation to create customizable datasets for various use cases.

## Features

- Interactive chat-based interface for dataset generation
- Customizable dataset parameters (size, resolution)
- Support for multiple AI models (Groq, OpenAI)
- Session management for maintaining context
- Sample image preview before full dataset generation
- RESTful API endpoints for integration

## Prerequisites

- Python 3.11+
- Node.js (for frontend development)
- Groq API key
- OpenAI API key
- Flask For backend API 

## Installation

1. Clone the repository
2. Install Python dependencies:
   ```bash
   pip install fastapi langchain langgraph langchain_groq langchain_openai python-dotenv
   ```
3. Install frontend dependencies:
   ```bash
   npm install
   ```

## Environment Setup

Create a `.env` file in the root directory with your API keys:

```env
GROQ_API_KEY=your_groq_api_key
OPENAI_API_KEY=your_openai_api_key
```

## API Endpoints

### POST /chat

Chat with the UI 
- Chat with the conversation Agent to seamlessly get preffered responses.
- You can Ask for sample images, review and fine tune the diversity of generated prompts which acts as the basis of the dataset.
- The Full Image Dataset Zip file will be Provided to the User once he is satisfied with sample images, prompts and is ready to get Full Dataset.

Process user messages and generate responses/images.
```json
{
    "message": "Generate images of mountains",
    "session_id": "unique_session_id"
}
```

### POST /set_dataset_parameters
Customize dataset generation parameters.
```json
{
    "num_images": 50,
    "resolution": "256x256",
    "session_id": "unique_session_id"
}
```

### POST /set_api_keys
Set custom API keys for external services.
```json
{
    "groq_api_key": "your_groq_api_key",
    "openai_api_key": "your_openai_api_key",
    "session_id": "unique_session_id"
}
```

### GET /sessions/{session_id}
Retrieve session information and status.

## Usage

1. Start the FastAPI server:
   ```bash
   python main.py
   ```
   The server will run on `http://localhost:8000`

2. Set up your API keys using the `/set_api_keys` endpoint

3. Start a conversation by sending a message to the `/chat` endpoint

4. Customize dataset parameters using `/set_dataset_parameters` if needed

5. Monitor session status and retrieve generated images through the session endpoint

## Project Structure

- `main.py`: FastAPI application and API endpoints
- `Dependancies.py`: Core functionality for image generation workflow
- `FS_BackendTesting.ipynb`: Development and testing notebook

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
