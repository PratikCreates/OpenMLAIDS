import os
import sys
import pandas as pd
import subprocess
import sqlite3
import json
import platform  # <--- NEW: To detect Windows
from dotenv import load_dotenv
from typing import Annotated
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import AzureChatOpenAI
from langchain_core.tools import tool
# NEW: Import SystemMessage to instruct the brain
from langchain_core.messages import SystemMessage 
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.sqlite import SqliteSaver
# NEW IMPORTS
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_experimental.utilities import PythonREPL
import matplotlib.pyplot as plt

# Judge Manager for self-evaluation
from src.judge_manager import judge_manager


# 1. Load Environment Variables
load_dotenv()

if not os.getenv("AZURE_OPENAI_API_KEY"):
    print("Error: AZURE_OPENAI_API_KEY not found in .env")
    sys.exit(1)

if not os.getenv("AZURE_OPENAI_ENDPOINT"):
    print("Error: AZURE_OPENAI_ENDPOINT not found in .env")
    sys.exit(1)

# 2. Define the State
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

# ---------------------------------------------------------
# TOOL DEFINITIONS
# ---------------------------------------------------------
@tool
def shell_tool(command: str):
    """Executes a shell command. Risk: HIGH."""
    try:
        # 10s timeout
        result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=10)
        output = result.stdout if result.returncode == 0 else result.stderr
        return output
    except Exception as e:
        return f"Shell Error: {str(e)}"

@tool
def download_kaggle_dataset(dataset_slug: str):
    """Downloads a dataset from Kaggle given the slug."""
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    download_path = "./data"
    os.makedirs(download_path, exist_ok=True)
    try:
        api.dataset_download_files(dataset_slug, path=download_path, unzip=True)
        files = os.listdir(download_path)
        return f"Files downloaded to {download_path}: {files}"
    except Exception as e:
        return f"Error downloading: {str(e)}"

@tool
def inspect_data(filename: str):
    """
    Reads the first 50 rows of a CSV file.
    Returns JSON strictly.
    """
    clean_filename = os.path.basename(filename)
    
    file_path = os.path.join("./data", clean_filename)
    if not os.path.exists(file_path):
        return json.dumps({"error": f"File {clean_filename} not found in ./data folder."})
    
    try:
        df = pd.read_csv(file_path)
        df_small = df.head(50)
        result = df_small.to_json(orient="split", index=False)
        return result
    except Exception as e:
        return json.dumps({"error": str(e)})

@tool
def python_helper(code: str):
    """
    Executes Python code. use this for data analysis, visualization, and complex calculations.
    Returns stdout/stderr.
    The environment has pandas as pd, matplotlib.pyplot as plt, numpy as np.
    To save a plot, use: plt.savefig('./static/filename.png')
    """
    repl = PythonREPL()
    try:
        # We wrap in try/except to catch runtime errors in the executing code
        result = repl.run(code)
        return f"Output:\n{result}"
    except Exception as e:
        return f"Execution Error: {str(e)}"

@tool
def web_search(query: str):
    """
    Searches the web using DuckDuckGo.
    Use this for finding documentation, libraries, or general knowledge.
    """
    search = DuckDuckGoSearchRun()
    return search.run(query)

@tool
def generate_report(content: str, filename: str = "REPORT.md"):
    """
    Generates a structured markdown report.
    Args:
        content: The full markdown content of the report.
        filename: Defaults to REPORT.md
    """
    with open(filename, "w", encoding="utf-8") as f:
        f.write(content)
    return f"Report saved to {filename}"

# NEW: Self-evaluation tool for agent improvement
@tool
def self_evaluate_agent(agent_response: str, context: str, judge_type: str = "general"):
    """
    Self-evaluates the agent's response using Oumi's LLM-as-a-Judge system.
    Returns detailed feedback and improvement suggestions.
    
    Args:
        agent_response: The agent's response to evaluate
        context: The user's original query/context
        judge_type: Type of judge to use ('general', 'data_science', 'problem_solving', 'code_quality')
    """
    try:
        # Use the Judge Manager to evaluate the response
        results = judge_manager.evaluate_agent_response(
            user_request=context,
            agent_response=agent_response,
            response_type=judge_type
        )
        
        # Get performance summary
        summary = judge_manager.get_performance_summary(results)
        
        return json.dumps({
            "evaluation_results": {name: {
                "score": result.score,
                "explanation": result.explanation,
                "detailed_feedback": result.detailed_feedback
            } for name, result in results.items()},
            "performance_summary": summary,
            "judge_type": judge_type,
            "status": "success"
        }, indent=2)
        
    except Exception as e:
        return json.dumps({
            "error": f"Self-evaluation failed: {str(e)}",
            "judge_type": judge_type,
            "status": "error"
        })

# NEW: Training data generation tool
@tool
def generate_training_data(prompt: str, agent_response: str, evaluation_result: str, output_file: str = "data/training/training_data.jsonl"):
    """
    Generates structured training data from agent interactions for future fine-tuning.
    
    Args:
        prompt: The original user prompt
        agent_response: The agent's response
        evaluation_result: The evaluation result from self_evaluate_agent
        output_file: Where to save the training data
    """
    try:
        # Parse evaluation result to extract scores
        eval_data = json.loads(evaluation_result)
        score = eval_data.get("evaluation", {}).get("score", 0)
        
        # Create training example
        training_example = {
            "instruction": prompt,
            "response": agent_response,
            "evaluation": {
                "score": score,
                "feedback": eval_data.get("evaluation", {}).get("feedback", ""),
                "judge_type": eval_data.get("judge_type", "general")
            },
            "metadata": {
                "timestamp": str(pd.Timestamp.now()),
                "model": "Azure-GPT-5.2",
                "framework": "OpenMLAIDS-v2.0"
            }
        }
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Append to JSONL file
        with open(output_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(training_example) + "\n")
            
        return f"Training data saved to {output_file}. Score: {score}/10"
        
    except Exception as e:
        return f"Error generating training data: {str(e)}"

# NEW: Model evaluation and performance tracking
@tool
def evaluate_model_performance(test_cases: str, model_name: str = "Azure-GPT-5.2", output_dir: str = "models/evaluation"):
    """
    Evaluates model performance using Oumi's evaluation framework.
    
    Args:
        test_cases: JSON string containing test cases or path to test file
        model_name: Name of the model being evaluated
        output_dir: Directory to save evaluation results
    """
    try:
        from src.fine_tuning_pipeline import create_evolution_pipeline
        
        # Parse test cases
        if os.path.exists(test_cases):
            with open(test_cases, 'r') as f:
                test_data = json.load(f)
        else:
            test_data = json.loads(test_cases)
        
        # Create evolution pipeline for evaluation
        pipeline = create_evolution_pipeline()
        
        # Run evaluation
        results = pipeline.evaluate_evolution(
            model_path=f"models/{model_name}",
            test_data=test_data
        )
        
        return json.dumps({
            "evaluation_results": results,
            "model_name": model_name,
            "status": "success",
            "output_dir": output_dir
        }, indent=2)
        
    except Exception as e:
        return f"Model evaluation failed: {str(e)}"

# NEW: Agent evolution tool
@tool
def evolve_agent(force_evolution: bool = False):
    """
    Triggers agent self-evolution using accumulated training data.
    Uses Oumi's fine-tuning capabilities to improve the agent.
    
    Args:
        force_evolution: Force evolution even if criteria aren't met
    """
    try:
        from src.fine_tuning_pipeline import create_evolution_pipeline
        
        pipeline = create_evolution_pipeline()
        
        # Check if evolution is recommended
        if not force_evolution and not pipeline.should_evolve():
            status = pipeline.get_evolution_status()
            return json.dumps({
                "status": "not_ready",
                "message": "Evolution not recommended yet. Need more high-quality training data.",
                "evolution_status": status
            }, indent=2)
        
        # Trigger evolution
        results = pipeline.evolve_agent()
        
        return json.dumps({
            "status": "success",
            "message": "Agent evolution completed successfully!",
            "evolution_results": results
        }, indent=2)
        
    except Exception as e:
        return json.dumps({
            "status": "error",
            "error": f"Evolution failed: {str(e)}"
        })

tools = [shell_tool, download_kaggle_dataset, inspect_data, python_helper, web_search, generate_report, self_evaluate_agent, generate_training_data, evaluate_model_performance, evolve_agent]

# ---------------------------------------------------------
# AZURE GPT-5.2 CONFIGURATION
# ---------------------------------------------------------
llm = AzureChatOpenAI(
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    extra_body={
        "reasoning_effort": "medium", 
        "verbosity": "medium",
        "max_completion_tokens": 4096
    }
)

llm_with_tools = llm.bind_tools(tools)

# ---------------------------------------------------------
# NODE LOGIC (UPDATED FOR OS AWARENESS + OUMI INTEGRATION)
# ---------------------------------------------------------
def chatbot_node(state: AgentState):
    # 1. Auto-Detect the OS
    current_os = platform.system()
    
    # 2. Create a System Instruction
    # This tells the Agent explicitly where it is and how to behave.
    system_instruction = SystemMessage(content=(
        f"SYSTEM CONTEXT: You are running on a {current_os} machine. "
        "You are a STATE-OF-THE-ART MACHINE LEARNING AGENT with self-evolution capabilities. "
        "Your goal is to perform end-to-end data science responsibly and autonomously.\n\n"
        "LIFECYCLE TO FOLLOW:\n"
        "1. EXPLORATION: Inspect data, check missing values, distributions. PLOT EVERYTHING using python_helper.\n"
        "2. PREPROCESSING: Clean data, impute, scale, encode. Use sklearn.\n"
        "3. MODELING: Use SOTA models (XGBoost, LightGBM, CatBoost) via python_helper. COMPARE baselines.\n"
        "4. TUNING: If needed, use Optuna for hyperparameter search.\n"
        "5. EVALUATION: Calculate metrics (AUC, F1, RMSE) and plots (Confusion Matrix, Feature Importance).\n"
        "6. REPORTING: Summarize findings and save a `REPORT.md` using `generate_report`.\n\n"
        "SELF-IMPROVEMENT CAPABILITIES:\n"
        "- Use `self_evaluate_agent` to critically assess your responses\n"
        "- Use `generate_training_data` to create learning examples from good responses\n"
        "- Use `evaluate_model_performance` to benchmark against standard datasets\n\n"
        "RULES:\n"
        "- PREFER `python_helper` for all logic/math.\n"
        "- Use `shell_tool` only for file ops.\n"
        f"- Shell commands must use {current_os.upper()} Syntax.\n"
        "- ALWAYS save plots as .png files.\n"
        "- If you need external libraries or info, use `web_search`.\n"
        "- Self-evaluate important responses for quality improvement\n"
        "- Generate training data from successful interactions\n"
    ))

    
    # 3. Prepend this instruction to the message history so the LLM sees it first
    messages = [system_instruction] + state["messages"]
    
    # 4. Invoke
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

# ---------------------------------------------------------
# GRAPH ARCHITECTURE
# ---------------------------------------------------------
workflow = StateGraph(AgentState)
workflow.add_node("chatbot", chatbot_node)
workflow.add_node("tools", ToolNode(tools))

workflow.add_edge(START, "chatbot")
workflow.add_conditional_edges("chatbot", tools_condition)
workflow.add_edge("tools", "chatbot")

conn = sqlite3.connect("openmlaids.db", check_same_thread=False)
memory = SqliteSaver(conn)

app = workflow.compile(
    checkpointer=memory,
    interrupt_before=["tools"]  # Safety switch - requires UI approval
)
