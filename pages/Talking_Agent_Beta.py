import streamlit as st
import random
import time
from utils.llm import get_location_coords, get_dates

class TalkingAgent:
    def __init__(self):
        self.data = {
            'latitude': None,
            'longitude': None,
            'start_date': None,
            'end_date': None
        }
        self.location_attempted = False  # Track if location has been attempted

    def process_input(self, user_input):
        # Try to parse location and date info from every user input
        self.get_location_info(user_input)
        self.get_date_info(user_input)
        update_sidebar()

    def get_location_info(self, user_input):
        if self.data['latitude'] is None and self.data['longitude'] is None:
            location = get_location_coords(user_input)
            print(location)
            if location and 'lat' in location.keys() and 'long' in location.keys():
                self.data['latitude'], self.data['longitude'] = location['lat'], location['long']
                message = "Got the location!"
                with st.chat_message("assistant"):
                    st.write(message)
                st.session_state.messages.append({"role": "assistant", "content": message})
                return message
            else:
                self.location_attempted = True  # Mark that location has been attempted
                message = "I couldn't determine the location. Please provide a time range (start and end dates)."
                with st.chat_message("assistant"):
                    st.write(message)
                st.session_state.messages.append({"role": "assistant", "content": message})
                return message

        return "I already have the location."

    def get_date_info(self, user_input):
        if self.data['start_date'] is None or self.data['end_date'] is None:
            dates = get_dates(user_input)
            print(dates)
            if dates and 'start' in dates.keys() and 'end' in dates.keys() and dates['start'] is not None and dates['end'] is not None:
                self.data['start_date'], self.data['end_date'] = dates['start'], dates['end']
                message = "Got the time range!"
                with st.chat_message("assistant"):
                    st.write(message)
                st.session_state.messages.append({"role": "assistant", "content": message})
                return message

        return "I already have the time range."

    def check_completion(self):
        # Check if all data points are filled
        return all(value is not None for value in self.data.values())

    def ask_for_missing_info(self):
        if self.data['latitude'] is None or self.data['longitude'] is None:
            if not self.location_attempted:
                message = "Please provide the location."
            else:
                message = "Please provide a time range (start and end dates)."
            with st.chat_message("assistant"):
                st.write(message)            
            st.session_state.messages.append({"role": "assistant", "content": message})
            return message
        elif self.data['start_date'] is None or self.data['end_date'] is None:
            message = "What time period are you interested in?"
            with st.chat_message("assistant"):
                st.write(message)
            st.session_state.messages.append({"role": "assistant", "content": message})
            return message
        return None  # If no missing info

    def conversation_flow(self, user_input):
        # Process the input first (try to get location and dates)
        self.process_input(user_input)
        
        # Check for missing information and ask the appropriate question
        if not self.check_completion():
            self.ask_for_missing_info()
        else:
            # If all inputs are gathered, proceed to API call or next step
            message = "All information gathered! Ready to make the API call."
            with st.chat_message("assistant"):
                st.write(message)
            st.session_state.messages.append({"role": "assistant", "content": message})
            self.get_sentinel_data()

    def get_sentinel_data(self):
        # Your API call logic goes here
        print(f"Fetching Sentinel data with {self.data}")
        message = f"Fetching Sentinel data with: {self.data}"
        with st.chat_message("assistant"):
            st.write(message)
        st.session_state.messages.append({"role": "assistant", "content": message})


# Initialize or retrieve the TalkingAgent instance from session state
if "agent" not in st.session_state:
    st.session_state.agent = TalkingAgent()

agent = st.session_state.agent

st.title("Information Retrieval Agent")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Update the sidebar as the value changes
def update_sidebar():
    with st.sidebar:
        st.write("Agent Data:")
        st.json(agent.data)

# Accept user input
if prompt := st.chat_input("Enter your query..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Process the input and run the conversation flow
    agent.conversation_flow(prompt)