from fastapi import FastAPI  # Import FastAPI framework for building APIs
from fastapi import Request  # Import Request class for handling requests
from fastapi.responses import JSONResponse  # Import JSONResponse class for constructing JSON responses
import quizzes  # Import module containing quiz data and answers
import time  # Import time module for time-related functions
from db_helper import insert_or_update_progress  # Import function for interacting with database to keep user scores
from API_helper import api_answer  # Import function for generating answers with Gemini


app = FastAPI()

# Define dictionaries to store quizzes and their answers
quizzes_dict = {
    "general_ml_quiz1": quizzes.general_ml_quiz1,
    "general_ml_quiz2": quizzes.general_ml_quiz2,
    "general_ml_quiz3": quizzes.general_ml_quiz3,
    "supervised_quiz1": quizzes.supervised_quiz1,
    "supervised_quiz2": quizzes.supervised_quiz2,
    "supervised_quiz3": quizzes.supervised_quiz3,
    "unsupervised_quiz1": quizzes.unsupervised_quiz1,
    "classification_quiz1": quizzes.classification_quiz1  # Removed extra space
}

quiz_answer_dict = {
    "answer_general_ml_quiz1": quizzes.answer_general_ml_quiz1,
    "answer_general_ml_quiz2": quizzes.answer_general_ml_quiz2,
    "answer_general_ml_quiz3": quizzes.answer_general_ml_quiz3,
    "answer_supervised_quiz1": quizzes.answer_supervised_quiz1,
    "answer_supervised_quiz2": quizzes.answer_supervised_quiz2,
    "answer_supervised_quiz3": quizzes.answer_supervised_quiz3,
    "answer_unsupervised_quiz1": quizzes.answer_unsupervised_quiz1,
    "answer_classification_quiz1": quizzes.answer_classification_quiz1
}

# Dictionaries to store user data
user_score = {}  # Store user scores
user_quizz = {}  # Store user quiz selections
user_answer = {}  # Store user answers
parent_intent = {}  # Store parent intents

# Handle incoming requests
@app.post("/")
async def handle_request(request: Request):
    # Retrieve the JSON data from the request
    payload = await request.json()
    question_id = None

    # Extract necessary information from the payload

    # Get username and intent from payload
    username = payload["originalDetectIntentRequest"]['payload']['data']['from']['id']
    intent = payload['queryResult']['intent']['displayName']
  
    # Get parameters and query text
    parameters = payload['queryResult']['parameters']
    query_text = payload['queryResult']['queryText']

    # Get output context
    output_context = payload['queryResult'].get('outputContexts', None)

    # Handling of Default Fallback Intent
    if intent == 'Default Fallback Intent':
        return fallback(output_context, query_text)

    # Save parent intent
    if intent in ["general_ml", "supervised", "unsupervised", "classification", "regression", "clustering", "dim_red"]:
        parent_intent[username] = intent

    # Choose the quiz
    if "quiz_choice" in parameters:
        user_quizz[username] = parent_intent[username] + "_" + parameters["quiz_choice"]
        user_answer[username] = "answer_" + parent_intent[username] + "_" + parameters["quiz_choice"]

    # Handling of questions
    if intent[:8] == 'question':
        question_id = intent[8:]
        return question(username, parameters, output_context, quizzes_dict[user_quizz[username]],
                        quiz_answer_dict[user_answer[username]], question_id, user_score)

    # Handling of score intent
    if intent == "score":
        return score(username, parameters, output_context,
                     quizzes_dict[user_quizz[username]],
                     quiz_answer_dict[user_answer[username]],
                     user_score)


# Function to handle scoring
def score(user, parameters: dict, output_context, quiz: dict, quiz_answer: dict, score_dict):
    # Construct active context
    prefix = output_context[0]['name'].split('/contexts/')[0] + '/contexts/'
    active_context = prefix + f"context_question9"

    if parameters["quiz-answer"] == quiz_answer["10"]:
        # Increment score if answer is correct
        score_dict[user] += 10
        fulfillment_text = f"Your score is {score_dict[user]}"
        insert_or_update_progress(user, user_quizz[user], score_dict[user])
    elif parameters["quiz-answer"] == 'next':
        fulfillment_text = f"Your score is {score_dict[user]}"
        insert_or_update_progress(user, user_quizz[user], score_dict[user])
    else:
        # Provide feedback if answer is incorrect
        fulfillment_text = api_answer(
            f"Question is:{quiz['9']} and my answer is {parameters['quiz-answer']} "
            f"explain why it is wrong and what is the correct answer.") + \
                           "\n\n Ask me other questions you may have or type 'next' if you want to see your score."
    return JSONResponse(content={
        "fulfillmentText": fulfillment_text,
        "outputContexts": [
            {
                "name": f"{active_context}",
                "lifespanCount": 2
            }]
    })


# Function to handle questions
def question(user, parameters: dict, output_context: list, quiz: dict, quiz_answer: dict, question_id, score):
    # Construct active context
    prefix = output_context[0]['name'].split('/contexts/')[0] + '/contexts/'
    active_context = prefix + f"context_question{question_id}"

    if question_id == "0":
        # Display first question and initialize score
        fulfillment_text = quiz[question_id]
        score[user] = 0
    elif parameters["quiz-answer"] == quiz_answer[question_id]:
        # Increment score if answer is correct
        fulfillment_text = "Correct answer!!! \n\n" + quiz[question_id]
        score[user] += 10
    elif parameters["quiz-answer"] == "next":
        # Display next question
        fulfillment_text = quiz[question_id]
    else:
        # Provide feedback if answer is incorrect
        fulfillment_text = api_answer(
            f"Question is:{quiz[str(int(question_id) - 1)]} and my answer is {parameters['quiz-answer']} explain why it is wrong and what is the correct answer") + \
                        " \n Ask me other questions you may have or type 'next' if you want to see the next question."
        active_context = prefix + f"context_question{int(question_id) - 1}"
        return JSONResponse(content={
            "fulfillmentText": fulfillment_text,
            "outputContexts": [
                {
                    "name": f"{active_context}",
                    "lifespanCount": 2
                }]
        })
 
    return JSONResponse(content={
            "outputContexts": [
                {
                    "name": active_context,
                    "lifespanCount": 2
                }],
            "fulfillmentMessages": [
                {
                    "payload": {
                        "telegram": {
                            "text": fulfillment_text,
                            "reply_markup": {
                                "keyboard": [
                                    ["a.", "b."],
                                    ["c.", "d."],
                                    ["Next"]
                                ],
                                "one_time_keyboard": True,
                                "resize_keyboard": True
                            }
                        }
                    }
                }
            ]
        }
        )


# Function to handle fallback intents
def fallback(context: list, query_text: str):
    # Generate response using API helper
    fulfillment_text = api_answer(query_text)
    print(fulfillment_text)
    return JSONResponse(content={
        "fulfillmentText": fulfillment_text,
        "outputContexts": [
            {
                "name": f"{context[0]['name']}",
                "lifespanCount": 2
            }
        ]
    })

