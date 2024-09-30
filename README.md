# Automating GeoAI Workflow


This project is designed to take user queries, utilize a large language model (LLM) agent to map those queries to the available foundation models, and display the results back to the user. By leveraging advanced AI capabilities, the system ensures accurate and efficient query processing, providing users with relevant and insightful results based on the detectors at hand.

# Setup and Run Instructions

1. **Create Conda Environment**
    ```sh
    conda env create -f environment.yml
    ```

2. **Create .env File**
    - Create a `.env` file in the root directory.
    - Add the following line to the `.env` file:
      ```
      GROQ_API_KEY=<your_key>
      ```

3. **Download Models**
    - Download models from [Hugging Face IBM NASA Geospatial](https://huggingface.co/ibm-nasa-geospatial).
    - Save the models to the `models` directory.
    - Save the configurations to the `configs` directory.

4. **Run the Application**
    ```sh
    streamlit run app.py
    ```