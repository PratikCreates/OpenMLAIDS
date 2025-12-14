# OpenMLAIDS - Open Machine Learning AI Data Science Assistant

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![OpenAI](https://img.shields.io/badge/Azure%20OpenAI-API-orange.svg)](https://azure.microsoft.com/en-us/services/cognitive-services/openai-service/)

**OpenMLAIDS** (Open Machine Learning AI Data Science Assistant) is an interactive CLI chatbot that helps non-technical professionals generate business insights without handling the complexity of machine learning or data science. It manages data cleaning, model training, code execution, and report generation end to end. With Oumi integration, OpenMLAIDS also supports reinforcement learning and model fine-tuning directly through the CLI.

> 🏆 **AI Agents Assemble Hackathon**: Built for the global "AI Agents Assemble" hackathon - a worldwide showdown where builders unite to create the next generation of intelligent agents. This project showcases how we assembled our skills, tools, and AI agents to democratize data science.

## 🎯 Key Features

- **No-Code Data Science**: Automate the entire ML pipeline from data preprocessing to model deployment
- **Business-Focused Insights**: Translates complex data science results into actionable business recommendations
- **Interactive CLI Interface**: Easy-to-use text-based interface for natural conversation
- **Self-Evolving AI**: Continuously improves performance through reinforcement learning and fine-tuning
- **Multi-Model Support**: Seamlessly switches between different AI models based on task complexity
- **Automated Reporting**: Generates professional reports with visualizations and insights
- **Kaggle Dataset Integration**: Seamlessly access and analyze thousands of public datasets
- **AI-Powered Development**: Built using Cline extension with intelligent subagent assistance

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
Create a `.env` file in the project root with your credentials:
```env
# Azure OpenAI Configuration
AZURE_OPENAI_API_KEY=your_api_key_here
AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT_NAME="gpt-5.2-chat"
AZURE_OPENAI_API_VERSION="2025-04-01-preview"

# Kaggle Integration (for dataset features)
KAGGLE_KEY=your_kaggle_key
KAGGLE_USERNAME=your_kaggle_username
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
- "Download the latest customer dataset from Kaggle and analyze it"
- "Fine-tune the model based on recent performance feedback"

## 🌟 What Makes OpenMLAIDS Special

### For Non-Technical Users
- **Zero Coding Required**: Simply describe what you want to analyze in plain English
- **Business-First Approach**: Results are presented as actionable business insights, not technical metrics
- **End-to-End Automation**: From data ingestion to final reports, everything happens automatically

### For Technical Users
- **Extensible Architecture**: Easy to add new tools, models, and capabilities
- **Self-Improving System**: Uses Oumi's reinforcement learning to get better over time
- **Advanced Integration**: Seamless connection with Kaggle, Azure OpenAI, and other platforms

### Hackathon Innovation
- **AI-Assisted Development**: Showcases the power of using Cline extension for rapid prototyping
- **Modern Tech Stack**: Combines cutting-edge frameworks like LangGraph with practical business applications
- **Real-World Impact**: Addresses the genuine need for accessible data science tools

## 🎬 Demo Video

[![OpenMLAIDS Demo](https://img.shields.io/badge/Watch-Demo%20Video-red?style=for-the-badge&logo=youtube)](https://drive.google.com/file/d/1KekFZ18dXk6ToBXBZ6CJrGQrORZq04mL/view?usp=sharing)

See OpenMLAIDS in action! The demo showcases real-time data analysis, model training, and business insight generation through natural language interactions.

## 🏗️ Tech Stack & Architecture

### Core Technologies
- **Python 3.11+**: Primary development language
- **LangChain & LangGraph**: Agent orchestration and conversation flow
- **Azure OpenAI GPT-5.2**: Advanced language model for natural language processing
- **Oumi Framework**: LLM-as-a-Judge evaluation and reinforcement learning
- **Textual**: Modern terminal user interface framework
- **Pandas & NumPy**: Data manipulation and analysis
- **Scikit-learn**: Machine learning algorithms and utilities
- **Matplotlib & Seaborn**: Data visualization and reporting

### Development Tools
- **Cline Extension**: AI-powered development assistant used throughout the project
- **Cline Subagents**: Specialized AI agents for different development tasks
- **Git**: Version control and collaboration

### Architecture Pattern
- **Agent-Based Design**: Modular agents handle specific tasks (data processing, model training, evaluation)
- **Event-Driven**: Reactive system responding to user inputs and tool outputs
- **Self-Improving**: Continuous learning through reinforcement feedback loops

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
- **Kaggle Integration**: Seamlessly downloads and works with public datasets from Kaggle
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
- `AZURE_OPENAI_DEPLOYMENT_NAME`: Your model deployment name (e.g., "gpt-5.2-chat")
- `AZURE_OPENAI_API_VERSION`: API version (e.g., "2025-04-01-preview")

### Kaggle Integration Setup
For dataset features, configure:
- `KAGGLE_KEY`: Your Kaggle API key
- `KAGGLE_USERNAME`: Your Kaggle username

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

## � ALearning & Growth

This project represents a journey in exploring cutting-edge AI technologies and their practical applications:

### Key Learning Areas
- **Agent-Based AI Systems**: Understanding how to orchestrate multiple AI agents for complex workflows
- **LLM Integration**: Practical experience with Azure OpenAI and advanced prompt engineering
- **Self-Improving Systems**: Implementation of reinforcement learning and continuous improvement loops
- **User Experience Design**: Creating intuitive CLI interfaces for complex technical processes
- **Data Science Automation**: Building end-to-end pipelines that require minimal human intervention

### Technical Growth
- **Advanced Python Development**: Leveraging modern frameworks like LangChain and Textual
- **AI-Assisted Development**: Extensive use of Cline extension and subagents for accelerated development
- **System Architecture**: Designing scalable, modular systems for AI applications
- **Performance Optimization**: Implementing efficient data processing and model management

### Business Impact Understanding
- **Democratizing AI**: Making advanced analytics accessible to non-technical users
- **Workflow Automation**: Reducing time-to-insight from days to minutes
- **Decision Support**: Translating complex data into actionable business recommendations

## 🙏 Acknowledgments

Special thanks to the "AI Agents Assemble" hackathon and the incredible tools that made this project possible:

- **[Cline](https://github.com/cline/cline)** 🤖: Thank you for the amazing AI-powered development assistant that accelerated our development process through intelligent code generation and subagent orchestration
- **[Oumi](https://github.com/oumi-ai/oumi)** 🌟: Thank you for providing the powerful LLM-as-a-Judge framework and reinforcement learning capabilities that enable OpenMLAIDS to continuously improve and self-evaluate
- Built with [LangChain](https://github.com/langchain-ai/langchain) and [LangGraph](https://github.com/langchain-ai/langgraph)
- Powered by [Azure OpenAI](https://azure.microsoft.com/en-us/services/cognitive-services/openai-service/)
- Inspired by the need to democratize data science and make AI accessible to everyone

---
*Making data science accessible to everyone, one conversation at a time.*
