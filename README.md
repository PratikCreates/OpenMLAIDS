# OpenMLAIDS

**ML + DS + AI = MLAIDS** - A conversational AI agent that democratizes data science through natural language interactions.

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![OpenAI](https://img.shields.io/badge/Azure%20OpenAI-API-orange.svg)](https://azure.microsoft.com/en-us/services/cognitive-services/openai-service/)

## What is OpenMLAIDS?

OpenMLAIDS transforms complex data science workflows into simple conversations. Users describe their analytical needs in plain English, and the system autonomously handles data acquisition, preprocessing, modeling, and insight generation.

**Built for the AI Agents Assemble hackathon** - showcasing next-generation intelligent agent orchestration.

## Core Capabilities

**Autonomous Data Science Pipeline:**
- Natural language query processing
- Automated dataset acquisition (Kaggle integration)
- Intelligent preprocessing and feature engineering
- Model selection and hyperparameter optimization
- Automated visualization and reporting

**Self-Evolving Intelligence:**
- Oumi-powered LLM-as-a-Judge evaluation system
- Reinforcement learning from user interactions
- Continuous model fine-tuning and improvement
- Performance tracking and optimization

**Enterprise-Grade Safety:**
- Tool execution approval system
- Sandboxed code execution environment
- Multi-judge validation framework
- Audit trail and performance monitoring

## Quick Start

```bash
git clone https://github.com/PratikCreates/OpenMLAIDS.git
cd OpenMLAIDS
python -m venv openmlaids_env
source openmlaids_env/bin/activate  # Windows: openmlaids_env\Scripts\activate
pip install -r requirements.txt
```

**Environment Configuration:**
```env
AZURE_OPENAI_API_KEY=your_api_key_here
AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT_NAME="gpt-5.2-chat"
AZURE_OPENAI_API_VERSION="2025-04-01-preview"
KAGGLE_KEY=your_kaggle_key
KAGGLE_USERNAME=your_kaggle_username
```

**Launch:**
```bash
python app.py
```

## Demo Video

[![Watch Demo](https://img.shields.io/badge/Watch-Demo%20Video-red?style=for-the-badge&logo=youtube)](https://drive.google.com/file/d/1KekFZ18dXk6ToBXBZ6CJrGQrORZq04mL/view?usp=sharing)

## Agent Tool System

OpenMLAIDS employs 10 specialized tools orchestrated through LangGraph:

**Data Acquisition & Processing:**
- `download_kaggle_dataset` - Automated dataset retrieval
- `inspect_data` - Intelligent data profiling and quality assessment
- `python_helper` - Sandboxed code execution for analysis

**Intelligence & Learning:**
- `self_evaluate_agent` - Multi-judge performance evaluation
- `generate_training_data` - Learning example creation
- `evaluate_model_performance` - Benchmarking and validation
- `evolve_agent` - Triggered self-improvement

**Integration & Output:**
- `web_search` - Real-time information gathering
- `generate_report` - Professional documentation
- `shell_tool` - System-level operations

**Safety Architecture:**
- Pre-execution tool approval workflow
- Multi-judge validation (Agent, Data Science, Problem Solving, Code Quality)
- Sandboxed execution environment
- Comprehensive audit logging

## Real-World Examples

**Customer Churn Analysis:**
```
User: "Analyze customer churn patterns and predict at-risk accounts"

Tool Calls:
1. download_kaggle_dataset("customer-churn-dataset")
2. inspect_data("churn_data.csv")
3. python_helper("# Feature engineering and model training")
4. self_evaluate_agent("response", "churn analysis", "data_science")
5. generate_report("Customer Churn Analysis Report")

Output: Predictive model + risk segmentation + retention strategies
```

**Sales Forecasting:**
```
User: "Forecast Q1 sales and identify growth opportunities"

Tool Calls:
1. inspect_data("sales_historical.csv")
2. web_search("Q1 2025 market trends retail")
3. python_helper("# Time series forecasting with external factors")
4. generate_training_data("forecast query", "model results", "evaluation")
5. generate_report("Q1 Sales Forecast Report")

Output: Time series model + growth projections + strategic recommendations
```

## Technical Architecture

**Core Framework:**
- **LangGraph** - Agent orchestration and workflow management
- **Azure OpenAI GPT-5.2** - Natural language understanding and generation
- **Oumi** - Self-evaluation and continuous improvement
- **Textual** - Interactive CLI interface

**Self-Evolution Pipeline:**
- Real-time response evaluation using 4 specialized judges
- Automated training data generation from successful interactions
- Fine-tuning pipeline for domain-specific optimization
- Performance tracking and evolution triggers

**Development Acceleration:**
- Built with Cline AI assistant for rapid prototyping
- AI-assisted code generation and optimization
- Intelligent subagent orchestration

## Project Structure

```
OpenMLAIDS/
├── app.py                    # Main CLI application
├── agent.py                  # Core agent with 10 specialized tools
├── src/
│   ├── judge_manager.py      # Oumi-powered evaluation system
│   ├── fine_tuning_pipeline.py # Self-evolution capabilities
│   └── performance_tracker.py  # Metrics and monitoring
├── configs/
│   ├── azure_config.yaml     # Azure OpenAI configuration
│   └── judge_configs/        # Multi-judge evaluation setup
└── data/                     # Dataset storage and training data
```

## Innovation Highlights

**Agent Orchestration:**
- Dynamic tool selection based on query complexity
- Multi-step reasoning with intermediate validation
- Autonomous error recovery and retry mechanisms

**Self-Improvement:**
- Oumi LLM-as-a-Judge for quality assessment
- Reinforcement learning from user feedback
- Automated fine-tuning based on performance metrics

**User Experience:**
- Zero-code data science for business users
- Real-time visualization and insight generation
- Professional report automation

## License

MIT License

## Acknowledgments

**AI Agents Assemble Hackathon** - Demonstrating the future of intelligent agent collaboration.

**Key Technologies:**
- **[Cline](https://github.com/cline/cline)** - AI-powered development acceleration
- **[Oumi](https://github.com/oumi-ai/oumi)** - Self-evaluation and improvement framework
- **LangChain/LangGraph** - Agent orchestration platform
- **Azure OpenAI** - Advanced language model capabilities

---
*Democratizing data science through conversational AI*