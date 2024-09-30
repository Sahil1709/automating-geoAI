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
