# Physician Matching Chat Agent 

## Introduction
------------
The  NAVI agent App is a Python application that allows you to communicate with the LLM-based chatbot. You can ask questions using natural language; the application will provide relevant responses.


## Dependencies and Installation
----------------------------
To install the ChatBot App, please follow these steps:

1. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Obtain an API key from OpenAI and add it to the `.env` file in the project directory.
```commandline
OPENAI_API_KEY=your_secrit_api_key
```

## Usage
-----
To use the ChatBot App, follow these steps:

1. Ensure that you have installed the required dependencies and added the OpenAI API key to the `.env` file.

2. Run the `main.py` file using the Streamlit CLI. Execute the following command:
   ```
   streamlit run app.py
   ```

3. The application will launch in your default web browser, displaying the user interface.

4. Ask questions in natural language using the chat interface.
