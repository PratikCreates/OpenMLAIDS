# OpenMLAIDS - Open Machine Learning AI Data Science Assistant

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![OpenAI](https://img.shields.io/badge/Azure%20OpenAI-API-orange.svg)](https://azure.microsoft.com/en-us/services/cognitive-services/openai-service/)

**OpenMLAIDS** (Open Machine Learning AI Data Science Assistant) is an interactive CLI chatbot tool designed to help non-technical professionals gain business insights by automating the machine learning and data science workflow. It handles data cleaning, model training, and report generation automatically, making advanced analytics accessible to everyone.

## 🎯 Key Features

- **No-Code Data Science**: Automate the entire ML pipeline from data preprocessing to model deployment
- **Business-Focused Insights**: Translates complex data science results into actionable business recommendations
- **Interactive CLI Interface**: Easy-to-use text-based interface for natural conversation
- **Self-Evolving AI**: Continuously improves performance through reinforcement learning and fine-tuning
- **Multi-Model Support**: Seamlessly switches between different AI models based on task complexity
- **Automated Reporting**: Generates professional reports with visualizations and insights

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Azure OpenAI API access
- Git (for cloning the repository)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/PratikCreates/OpenMLAIDS.git
cd OpenMLAIDS
```

2. Create a virtual environment:
```bash
python -m venv openmlaids_env
source openmlaids_env/bin/activate  # On Windows: openmlaids_env\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure environment variables:
Create a `.env` file in the project root with your Azure OpenAI credentials:
```env
AZURE_OPENAI_API_KEY=your_api_key_here
AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT_NAME=your_deployment_name
AZURE_OPENAI_API_VERSION=2024-12-01-preview
```

### Usage

Run the application:
```bash
python app.py
```

Once launched, you can interact with OpenMLAIDS through natural language commands like:
- "Analyze sales data and predict future trends"
- "What factors influence customer churn?"
- "Create a report on marketing campaign effectiveness"

## 🧠 How It Works

OpenMLAIDS combines several advanced technologies to provide a seamless data science experience:

### Core Architecture
1. **LangGraph Framework**: Manages conversation flow and tool orchestration
2. **Oumi Integration**: Provides LLM-as-a-Judge evaluation and self-improvement capabilities
3. **Dynamic Model Selection**: Automatically chooses the optimal model for each task
4. **Continuous Learning Loop**: Evolves performance through reinforcement learning

### Workflow Process
1. **Task Analysis**: Determines complexity and requirements of user requests
2. **Data Processing**: Handles data cleaning, transformation, and feature engineering
3. **Model Development**: Selects and trains appropriate ML models
4. **Evaluation & Validation**: Assesses model performance using multiple metrics
5. **Insight Generation**: Converts results into business-relevant insights
6. **Report Creation**: Produces professional documentation with visualizations

## 🛠️ Technical Capabilities

### Supported Tasks
- Predictive modeling and forecasting
- Customer segmentation and clustering
- A/B testing analysis
- Marketing attribution modeling
- Risk assessment and fraud detection
- Sales and demand forecasting
- Text analysis and sentiment mining

### Integration Features
- **Web Search**: Automatically researches relevant information
- **Code Execution**: Safely executes Python code for analysis
- **Data Visualization**: Creates charts and graphs for better understanding
- **Kaggle Integration**: Downloads and works with public datasets
- **File Operations**: Handles data import/export seamlessly

## 📈 Self-Evolution System

OpenMLAIDS continuously improves through:
- **Performance Monitoring**: Tracks success rates and quality scores
- **Reinforcement Learning**: Learns from successful interactions
- **Model Fine-tuning**: Adapts to domain-specific requirements
- **Judged Feedback**: Uses LLM-as-a-Judge for quality assessment

## 📊 Example Use Cases

### Business Intelligence
```
User: "Analyze our Q4 sales data and identify growth opportunities"
OpenMLAIDS: [Processes data, identifies trends, creates visualizations, and provides strategic recommendations]
```

### Predictive Analytics
```
User: "Predict which customers are likely to churn next quarter"
OpenMLAIDS: [Builds churn prediction model, identifies at-risk customers, suggests retention strategies]
```

### Market Research
```
User: "Research competitors in our industry and summarize key findings"
OpenMLAIDS: [Searches web, analyzes data, creates competitive landscape report]
```

## 📁 Project Structure

```
OpenMLAIDS/
├── app.py                 # Main application entry point
├── agent.py               # Core agent logic and tools
├── requirements.txt       # Python dependencies
├── configs/               # Configuration files
│   ├── azure_config.yaml  # Azure OpenAI settings
│   └── judge_configs/     # Oumi judge configurations
├── data/                  # Sample datasets
│   └── train_and_test2.csv # Example Titanic dataset
├── src/                   # Core modules
│   ├── judge_manager.py        # LLM-as-a-Judge system
│   ├── model_manager.py        # Dynamic model selection
│   ├── fine_tuning_pipeline.py # Self-evolution capabilities
│   ├── continuous_improvement_loop.py # Learning system
│   └── performance_tracker.py  # Monitoring and analytics
└── models/                # Trained models and evolution data
```

## 🔧 Configuration

### Azure OpenAI Setup
Ensure you have the following environment variables configured:
- `AZURE_OPENAI_API_KEY`: Your Azure OpenAI API key
- `AZURE_OPENAI_ENDPOINT`: Your Azure OpenAI endpoint URL
- `AZURE_OPENAI_DEPLOYMENT_NAME`: Your model deployment name
- `AZURE_OPENAI_API_VERSION`: API version (default: 2024-12-01-preview)

### Judge Configurations
The system uses four specialized judges for self-evaluation:
- **Agent Evaluation Judge**: General performance assessment
- **Data Science Judge**: Technical accuracy evaluation
- **Problem Solving Judge**: Solution quality assessment
- **Code Quality Judge**: Programming standards review

## 🤝 Contributing

We welcome contributions to OpenMLAIDS! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details on how to get involved.

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with [LangChain](https://github.com/langchain-ai/langchain) and [LangGraph](https://github.com/langchain-ai/langgraph)
- Powered by [Azure OpenAI](https://azure.microsoft.com/en-us/services/cognitive-services/openai-service/)
- Enhanced with [Oumi](https://github.com/oumi-ai/oumi) for self-evaluation capabilities
- Inspired by the need to democratize data science

## 📞 Support

For support, please open an issue on GitHub or contact the development team.

---
*Making data science accessible to everyone, one conversation at a time.*
