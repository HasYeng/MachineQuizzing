import google.generativeai as genai  # Import the generative AI module
import re  # Import the regular expression module

# Function to remove markdown syntax from text
def remove_markdown(text):
    # Remove bold and italic syntax
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    text = re.sub(r'\*(.*?)\*', r'\1', text)

    # Remove bullet points
    text = re.sub(r'[\*\-] ', '', text)

    return text

# Set up Google API key and safety settings
GOOGLE_API_KEY = 'YOUR API KEY'  # Insert your Google API key here
safety_settings = [
    {"category": "HARM_CATEGORY_DANGEROUS", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
]

# Configure Generative AI with API key and safety settings
genai.configure(api_key=GOOGLE_API_KEY)

# Initialize GenerativeModel for chat
model = genai.GenerativeModel(model_name='gemini-pro', safety_settings=safety_settings)

# Start a chat session
chat = model.start_chat(history=[])

# Send introductory messages to the chat
chat.send_message("""When asked questions keep your answers as short as possible and don't give any extra information unless specifically asked.
Please don't use any formatting like markdown answer with plain text, do not use *s or bold text or bluepoints""")
chat.send_message(
    """Picture yourself as my trusted ML advisor, I will ask you ML related questions please answer them. 
    I'll also give you quizzes and the answer options, my answer which will likely be wrong,
    don't give me the quiz answers just give me direction. 
    I'm turning to you for assistance with my quizzes or any ML inquiries. Offer insights, hints, 
    or answers to help me progress. Please help me learn.""")

# Function to generate AI answer for a given question
def api_answer(question):
    response = chat.send_message(question).text  # Get AI response from the chat model
    return remove_markdown(response)  # Remove markdown syntax from the response


