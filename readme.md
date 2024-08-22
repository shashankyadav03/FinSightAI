
# FinSightAI: Advanced AI Financial Advice with LLaMA 8B and Streamlit

## Project Overview
FinSightAI is at the forefront of combining advanced AI technology with user-friendly applications to deliver personalized financial advice. Based on the LLaMA 8B Instruct Model and enhanced through LoRA (Low-Rank Adaptation), this project integrates robust AI capabilities into a Streamlit app to offer real-time, tailored financial recommendations, focusing on asset allocation strategies using the finance-specific 'finance-alpaca' dataset.

## Key Features
- **Personalised Financial Recommendations**: Utilizes fine-tuned LLaMA 8B model to generate asset allocation advice specific to user inputs.
- **Efficient AI Modeling**: Incorporates LoRA fine-tuning for optimal balance between computational efficiency and deep learning performance.
- **Interactive Streamlit Application**: Provides a conversational interface for users to interact with the AI, ask financial questions, and receive instant advice.
- **Real-Time Data Verification**: Enhances trust by allowing users to verify AI-generated advice against real-time market data.

## Detailed Components

### Model Architecture and Fine-Tuning
- **Base Model**: LLaMA 8B, selected for its efficiency and effectiveness in handling specific financial recommendation tasks.
- **Fine-Tuning Methodology**:
  - **Dataset**: Uses finance-alpaca dataset, aimed at instruction-tuning within financial contexts.
  - **LoRA Parameters**:
    - **Rank**: 32
    - **Alpha**: 16
    - **Target Modules**: `q_proj` and `v_proj` in the transformer layers to adapt to finance-specific tasks.
  - **Quantization**: Implements QLoRA (4-bit) to minimize memory usage and enhance inference speed.

### Streamlit Application Setup
- **User Interaction**: Simulates a financial advisory chat where users can input their financial documents and queries.
- **Real-Time Inference**: Generates advice based on the user’s financial data and questions in real-time.
- **Market News Verification**: Includes functionality to check AI advice against the latest market news through an integrated news API.

### Technical Setup and Deployment
- **Key Technologies**: Python, `transformers`, `torch`, and `streamlit`.
- **Inference Pipeline**:
  - Tokenization of user inputs and response generation using the fine-tuned model.
  - Real-time response verification using external news sources.
- **Deployment Options**: Can be run locally or deployed on cloud platforms like Hugging Face Spaces, with the model saved using LoRA for efficient storage and retrieval.

### Example Use Case Scenario
- A user uploads a financial report in the Streamlit app, seeking advice on portfolio diversification.
- The system analyzes the report using the fine-tuned model and proposes an asset allocation strategy.
- The user verifies this strategy with current market trends provided via the app’s news verification feature and adjusts based on the AI’s recommendations.

## Getting Started
Detailed instructions on cloning the repository, setting up the environment, and launching the Streamlit application are provided to help users quickly set up and start using FinSightAI.

## Running the Streamlit Application

To start using the FinSightAI Streamlit application locally, follow these steps:

1. **Set Up Your Environment**:
   Ensure that you have already cloned the repository and installed all dependencies as outlined in the "Getting Started" section.

2. **Navigate to the Application Directory**:
   Open a terminal and change into the project directory where the Streamlit script is located:
   ```bash
   cd FinSightAI/
   ```

3. **Launch the Streamlit App**:
   Run the following command to start the Streamlit server:
   ```bash
   streamlit run main.py
   ```

   This command will start the Streamlit server and the app will automatically open in your default web browser. If it does not open automatically, you can manually navigate to `http://localhost:8501` in your browser to view the application.

4. **Using the App**:
   - **Upload a Financial Document**: Click on the upload area to add your financial documents to the application. Supported formats include PDF, CSV, and Excel files.
   - **Enter Financial Questions**: Use the text input box to ask specific financial questions or seek advice on asset allocation.
   - **View Recommendations and Verify**: Once the data is processed, the AI model will provide recommendations. Use the 'Verify' button to compare these against real-time market data from integrated news sources.

5. **Troubleshooting**:
   - If you encounter any issues with the app loading or functioning correctly, make sure all environment variables and dependencies are set correctly.
   - Check the console for any error messages that might indicate what went wrong and adjust your setup accordingly.

## Contributing
We encourage contributions from the community. Check our [Contributing Guide](CONTRIBUTING.md) for how to participate in improving FinSightAI.

## License
This project is licensed under the MIT License.

## Acknowledgements
- Special thanks to all contributors and community members who have made this project possible through their insights and feedback.
