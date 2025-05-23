# Ennchan RAG Proof of Concept

A locally hosted chatbot that answers questions using information from the internet.

## Project Overview

### What is this?
This is a Q&A chatbot that receives user questions, searches the internet for answers, and provides summarized responses. The primary goal was to create a local chatbot that runs directly on your PC.

### Why build it?
This project addresses the limitations of services like ChatGPT and DeepSeek, which often face usage limits and server congestion. By hosting the chatbot locally, users can maintain consistent access and prompt quality while having more control over the system.

## Features

### Core Functionality
- Local-first architecture
- Interactive Q&A capabilities
- Customizable model selection
- Configurable search engine integration (planned)

### Query Capabilities
Users can ask any type of question, and the chatbot will provide answers based on the capabilities of the loaded model.

### How it Works
1. The system processes the user's prompt and converts it into a search query
2. Sends the query to the search engine API
3. Compiles the search results
4. Uses information retrieval strategies to select relevant context
5. Triggers the LLM to generate a response based on the selected context

## Optimizations

### Performance Improvements
- Implemented multiprocessing for better resource utilization
- Added network retry mechanisms for improved reliability

## Interfaces

### Current
- Command Line Interface (CLI)

### Coming Soon
- Web App API
