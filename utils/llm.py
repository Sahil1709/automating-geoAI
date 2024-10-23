from groq import Groq
from dotenv import load_dotenv
import os
import json
load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY"),
)

def get_detectors():
    detector_list = [file.split('.')[0] for file in os.listdir('models')]
    return detector_list

def get_completion(user_input):
    completion = client.chat.completions.create(
        model="llama-3.2-11b-text-preview",
        messages=[
            {
                "role": "system",
                "content": f'''
                You need to map user query to the list of available detectors. \navaialble detector = {get_detectors()}\n
                \nYour JSON schema should follow the following syntax\n{{\n\"detector\": \"name of one of the detectors from list\"\n}}
                If the detector is not in the list return {{\n\"detector\": null\n}}
                '''
            },
            {
                "role": "user",
                "content": user_input
            }
        ],
        temperature=1,
        max_tokens=1024,
        top_p=1,
        stream=False,
        response_format={"type": "json_object"},
        stop=None,
    )

    print(completion.choices[0].message)
    return json.loads(completion.choices[0].message.content)

def get_location_coords(user_input):
    completion = client.chat.completions.create(
        model="llama-3.1-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": "You're a location coordinates giver. Based on the user input your task is to identify the place present in the user input and just give me the coordinates of that place/location/area and ignore everything else. Get that locations coordinates and return your JSON response in the following schema:\n\n{\n    \"lat\":\n    \"long\":\n}"
            },
            {
                "role": "user",
                "content": user_input
            }
        ],
        temperature=0,
        max_tokens=1024,
        top_p=1,
        stream=False,
        response_format={"type": "json_object"},
        stop=None,
    )

    print(completion.choices[0].message)
    return json.loads(completion.choices[0].message.content)

def get_dates(user_input):
    completion = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[
            {
                "role": "system",
                "content": "You're a date parser. You need to identify dates from user query and convert it into a form of yyyy-mm-dd. If you are not able to find any dates return null in start and end. Your JSON schema should follow the following:\n\n{\n    \"start\": yyyy-mm-dd,\n    \"end\": yyyy-mm-dd\n}"
            },
            {
                "role": "user",
                "content": user_input
            }
        ],
        temperature=0,
        max_tokens=1024,
        top_p=1,
        stream=False,
        response_format={"type": "json_object"},
        stop=None,
    )

    print(completion.choices[0].message)
    return json.loads(completion.choices[0].message.content)