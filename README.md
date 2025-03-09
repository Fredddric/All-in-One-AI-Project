# DeepSeek AI Final Year Project

This project demonstrates the integration of the DeepSeek AI API into a Python application, showcasing text generation and image analysis capabilities.

## Features

- **Text Generation**: Generate AI responses using DeepSeek's powerful language models
- **Image Analysis**: Upload and analyze images with AI vision capabilities
- **User-friendly Interface**: Built with Streamlit for easy interaction

## Setup Instructions

1. Clone or download this repository to your local machine

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run the Streamlit application:
   ```
   streamlit run app.py
   ```

4. Access the application through your web browser at `http://localhost:8501`

## Project Structure

- `app.py`: Main application with Streamlit UI and DeepSeek API integration
- `requirements.txt`: List of Python dependencies
- `.env`: Environment file containing your DeepSeek API key

## Important Notes

- Your DeepSeek API key is stored in the `.env` file - keep this file secure
- The application requires an internet connection to communicate with the DeepSeek API
- The image analysis feature works with JPG, JPEG, and PNG formats

## Future Enhancements

- Adding more DeepSeek AI capabilities as they become available
- Implementing conversation history for text generation
- Adding export functionality for AI-generated content
- Creating a more advanced model fine-tuning section
