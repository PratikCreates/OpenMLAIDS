# -*- coding: utf-8 -*-
"""
OpenMLAIDS + Oumi Dynamic Model Selection System
Intelligent model switching based on task complexity and performance
"""

import os
import json
import time
import yaml
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

class TaskComplexity(Enum):
    """Task complexity levels"""
    SIMPLE = "simple"
    MODERATE = "moderate" 
    COMPLEX = "complex"
    EXPERT = "expert"

class ModelType(Enum):
    """Available model types"""
    AZURE_GPT5_2 = "azure_gpt5_2"
    LOCAL_SLM_360M = "local_slm_360m"
    LOCAL_SLM_1_7B = "local_slm_1_7b"
    EVOLVED_MODEL = "evolved_model"

@dataclass
class ModelCapability:
    """Model capabilities and characteristics"""
    name: str
    model_type: ModelType
    context_length: int
    max_tokens: int
    speed: float  # tokens per second
    quality_score: float  # 0-10
    cost_per_token: float
    strengths: List[str]
    weaknesses: List[str]
    specializations: List[str]
    
    def to_dict(self) -> Dict:
        return asdict(self)

@dataclass
class TaskProfile:
    """Profile of a task to be completed"""
    complexity: TaskComplexity
    domain: str  # e.g., "data_science", "code", "general"
    expected_output_length: int
    requires_reasoning: bool
    requires_coding: bool
    requires_analysis: bool
    time_sensitivity: str  # "fast", "normal", "thorough"
    
    def to_dict(self) -> Dict:
        return {
            "complexity": self.complexity.value,
            "domain": self.domain,
            "expected_output_length": self.expected_output_length,
            "requires_reasoning": self.requires_reasoning,
            "requires_coding": self.requires_coding,
            "requires_analysis": self.requires_analysis,
            "time_sensitivity": self.time_sensitivity
        }

class DynamicModelManager:
    """
    Intelligent model selection and switching system
    """
    
    def __init__(self, config_path: str = "configs/azure_config.yaml"):
        """Initialize the model manager"""
        self.config_path = config_path
        self.models_dir = "models"
        self.performance_dir = f"{self.models_dir}/performance"
        
        # Create directories
        Path(self.models_dir).mkdir(parents=True, exist_ok=True)
        Path(self.performance_dir).mkdir(parents=True, exist_ok=True)
        
        # Initialize model registry
        self.models = self._initialize_models()
        
        # Performance tracking
        self.performance_history = {}
        self.current_model = None
        self.task_queue = []
        
        # Load performance history
        self._load_performance_history()
        
    def _initialize_models(self) -> Dict[str, ModelCapability]:
        """Initialize available models with their capabilities"""
        models = {}
        
        # Azure GPT-5.2 (Premium model)
        models["azure_gpt5_2"] = ModelCapability(
            name="Azure OpenAI GPT-5.2",
            model_type=ModelType.AZURE_GPT5_2,
            context_length=128000,
            max_tokens=4096,
            speed=50.0,  # tokens per second
            quality_score=9.5,
            cost_per_token=0.002,
            strengths=["complex reasoning", "code generation", "data analysis", "creative writing"],
            weaknesses=["costly", "rate limited"],
            specializations=["data_science", "advanced coding", "complex problem solving"]
        )
        
        # Local SmolLM-360M (Efficient model)
        models["local_slm_360m"] = ModelCapability(
            name="SmolLM-360M-Instruct",
            model_type=ModelType.LOCAL_SLM_360M,
            context_length=2048,
            max_tokens=1024,
            speed=200.0,  # tokens per second
            quality_score=7.2,
            cost_per_token=0.0,
            strengths=["fast responses", "simple tasks", "basic analysis"],
            weaknesses=["limited reasoning", "short context"],
            specializations=["simple queries", "quick data exploration"]
        )
        
        # Local SmolLM-1.7B (Balanced model)
        models["local_slm_1_7b"] = ModelCapability(
            name="SmolLM-1.7B-Instruct", 
            model_type=ModelType.LOCAL_SLM_1_7B,
            context_length=4096,
            max_tokens=2048,
            speed=100.0,  # tokens per second
            quality_score=8.1,
            cost_per_token=0.0,
            strengths=["balanced performance", "good reasoning", "moderate complexity"],
            weaknesses=["still limited vs premium models"],
            specializations=["moderate complexity tasks", "general purpose"]
        )
        
        # Evolved models (dynamic based on training)
        models["evolved_model"] = ModelCapability(
            name="OpenMLAIDS Evolved Model",
            model_type=ModelType.EVOLVED_MODEL,
            context_length=4096,
            max_tokens=2048,
            speed=80.0,  # tokens per second
            quality_score=8.5,  # Will improve over time
            cost_per_token=0.0,
            strengths=["specialized training", "optimized performance"],
            weaknesses=["limited to trained domains"],
            specializations=["data science workflows", "agent interactions"]
        )
        
        return models
    
    def analyze_task(self, task_description: str, user_input: str = None) -> TaskProfile:
        """
        Analyze a task to determine its profile and complexity
        
        Args:
            task_description: Description of the task
            user_input: Additional context from user
            
        Returns:
            TaskProfile object
        """
        task_lower = task_description.lower()
        
        # Determine complexity
        complexity_indicators = {
            TaskComplexity.SIMPLE: [
                "what is", "define", "explain", "simple", "basic", "quick"
            ],
            TaskComplexity.MODERATE: [
                "analyze", "compare", "create", "generate", "build", "implement"
            ],
            TaskComplexity.COMPLEX: [
                "optimize", "comprehensive", "detailed analysis", "machine learning", 
                "model", "algorithm", "predict", "forecast"
            ],
            TaskComplexity.EXPERT: [
                "research", "investigate", "develop", "architect", "design system",
                "end-to-end", "production", "enterprise", "cutting-edge"
            ]
        }
        
        # Calculate complexity score
        complexity_scores = {level: 0 for level in TaskComplexity}
        for level, indicators in complexity_indicators.items():
            for indicator in indicators:
                if indicator in task_lower:
                    complexity_scores[level] += 1
        
        # Determine complexity level
        max_score = max(complexity_scores.values())
        if max_score == 0:
            complexity = TaskComplexity.MODERATE
        else:
            complexity = max(complexity_scores, key=complexity_scores.get)
        
        # Determine domain
        domain_indicators = {
            "data_science": ["data", "analysis", "statistics", "machine learning", "model", "predict"],
            "code": ["code", "programming", "python", "function", "algorithm", "implement"],
            "visualization": ["chart", "plot", "graph", "visualize", "dashboard"],
            "research": ["research", "study", "investigate", "paper", "literature"],
            "general": []  # default
        }
        
        domain = "general"
        for domain_name, indicators in domain_indicators.items():
            if any(indicator in task_lower for indicator in indicators):
                domain = domain_name
                break
        
        # Determine requirements
        requires_reasoning = any(word in task_lower for word in [
            "why", "how", "because", "reason", "logic", "analyze", "think"
        ])
        
        requires_coding = any(word in task_lower for word in [
            "code", "program", "function", "script", "implement", "python"
        ])
        
        requires_analysis = any(word in task_lower for word in [
            "analyze", "examine", "investigate", "compare", "evaluate"
        ])
        
        # Estimate output length
        output_length = 500  # default
        if "comprehensive" in task_lower or "detailed" in task_lower:
            output_length = 2000
        elif "brief" in task_lower or "quick" in task_lower:
            output_length = 200
        
        # Determine time sensitivity
        time_sensitivity = "normal"
        if "urgent" in task_lower or "quickly" in task_lower:
            time_sensitivity = "fast"
        elif "thorough" in task_lower or "comprehensive" in task_lower:
            time_sensitivity = "thorough"
        
        return TaskProfile(
            complexity=complexity,
            domain=domain,
            expected_output_length=output_length,
            requires_reasoning=requires_reasoning,
            requires_coding=requires_coding,
            requires_analysis=requires_analysis,
            time_sensitivity=time_sensitivity
        )
    
    def select_optimal_model(self, task_profile: TaskProfile) -> Tuple[str, float]:
        """
        Select the optimal model for a given task
        
        Args:
            task_profile: Profile of the task to be completed
            
        Returns:
            Tuple of (model_name, confidence_score)
        """
        available_models = list(self.models.keys())
        
        # Calculate scores for each model
        model_scores = {}
        
        for model_name, model_cap in self.models.items():
            score = 0.0
            
            # Base quality score
            score += model_cap.quality_score * 0.3
            
            # Speed score (inverted for slower models)
            speed_score = min(model_cap.speed / 100.0, 1.0)  # Normalize to 0-1
            if task_profile.time_sensitivity == "fast":
                score += speed_score * 0.25
            else:
                score += speed_score * 0.15
            
            # Complexity matching
            complexity_scores = {
                TaskComplexity.SIMPLE: [model_cap.quality_score * 0.8],  # Good enough for simple
                TaskComplexity.MODERATE: [model_cap.quality_score * 0.9],
                TaskComplexity.COMPLEX: [model_cap.quality_score * 1.1 if model_cap.quality_score >= 8.0 else 0],
                TaskComplexity.EXPERT: [model_cap.quality_score * 1.2 if model_cap.quality_score >= 9.0 else 0]
            }
            score += complexity_scores[task_profile.complexity][0] * 0.25
            
            # Domain specialization
            domain_bonus = 0
            if task_profile.domain in model_cap.specializations:
                domain_bonus += 2.0
            score += domain_bonus * 0.1
            
            # Cost consideration
            if task_profile.complexity in [TaskComplexity.SIMPLE, TaskComplexity.MODERATE]:
                if model_cap.cost_per_token == 0:  # Free local models preferred for simple tasks
                    score += 1.0
            else:  # For complex tasks, willing to pay for quality
                score += model_cap.quality_score * 0.1
            
            # Context length requirements
            if task_profile.expected_output_length > model_cap.max_tokens:
                score *= 0.5  # Penalty for insufficient context
            
            # Reasoning requirements
            if task_profile.requires_reasoning and model_cap.quality_score < 8.0:
                score *= 0.7  # Penalty for weak reasoning
            
            model_scores[model_name] = score
        
        # Select best model
        best_model = max(model_scores, key=model_scores.get)
        confidence = model_scores[best_model] / 10.0  # Normalize to 0-1
        
        # Log selection
        print(f"🎯 Model Selection: {best_model} (confidence: {confidence:.2f})")
        print(f"   Task: {task_profile.complexity.value} complexity, {task_profile.domain} domain")
        print(f"   Reason: {self._explain_selection(best_model, task_profile)}")
        
        return best_model, confidence
    
    def _explain_selection(self, model_name: str, task_profile: TaskProfile) -> str:
        """Explain why a model was selected"""
        model_cap = self.models[model_name]
        reasons = []
        
        if model_cap.quality_score >= 9.0:
            reasons.append("high quality")
        if task_profile.time_sensitivity == "fast" and model_cap.speed >= 100:
            reasons.append("fast processing")
        if task_profile.complexity in [TaskComplexity.COMPLEX, TaskComplexity.EXPERT]:
            reasons.append("handles complex tasks")
        if task_profile.domain in model_cap.specializations:
            reasons.append("domain specialization")
        if model_cap.cost_per_token == 0 and task_profile.complexity in [TaskComplexity.SIMPLE, TaskComplexity.MODERATE]:
            reasons.append("cost-effective for simple tasks")
        
        return ", ".join(reasons) or "balanced performance"
    
    def execute_with_model(self, task_description: str, model_name: str, **kwargs) -> Dict[str, Any]:
        """
        Execute a task with a specific model
        
        Args:
            task_description: Description of the task
            model_name: Name of the model to use
            **kwargs: Additional parameters for the model
            
        Returns:
            Execution results
        """
        start_time = time.time()
        
        try:
            # Get model capabilities
            if model_name not in self.models:
                return {"status": "error", "error": f"Model {model_name} not found"}
            
            model_cap = self.models[model_name]
            
            # Execute with the selected model
            if model_name == "azure_gpt5_2":
                result = self._execute_with_azure_gpt5_2(task_description, **kwargs)
            elif model_name.startswith("local_slm"):
                result = self._execute_with_local_model(task_description, model_name, **kwargs)
            elif model_name == "evolved_model":
                result = self._execute_with_evolved_model(task_description, **kwargs)
            else:
                result = {"status": "error", "error": f"Execution method not implemented for {model_name}"}
            
            # Record performance
            execution_time = time.time() - start_time
            self._record_performance(
                model_name=model_name,
                task_description=task_description,
                result=result,
                execution_time=execution_time
            )
            
            # Add metadata
            result["execution_metadata"] = {
                "model_used": model_name,
                "model_capabilities": model_cap.to_dict(),
                "execution_time": execution_time,
                "timestamp": datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            return {
                "status": "error", 
                "error": str(e),
                "model_name": model_name,
                "execution_time": time.time() - start_time
            }
    
    def _execute_with_azure_gpt5_2(self, task_description: str, **kwargs) -> Dict[str, Any]:
        """Execute task with Azure GPT-5.2"""
        try:
            # Use the existing agent system
            from agent import app as agent_app
            
            config = {"configurable": {"thread_id": "model_selection_session"}}
            input_message = {"messages": [("user", task_description)]}
            
            results = []
            for event in agent_app.stream(input_message, config=config):
                results.append(event)
            
            return {
                "status": "success",
                "model": "azure_gpt5_2",
                "results": results,
                "response": "Task completed with Azure GPT-5.2"
            }
            
        except Exception as e:
            return {"status": "error", "error": f"Azure GPT-5.2 execution failed: {str(e)}"}
    
    def _execute_with_local_model(self, task_description: str, model_name: str, **kwargs) -> Dict[str, Any]:
        """Execute task with local model (mock implementation)"""
        # Mock local model execution
        time.sleep(1)  # Simulate processing time
        
        return {
            "status": "success",
            "model": model_name,
            "response": f"Task completed with {model_name} (mock execution)",
            "note": "This is a mock implementation. Replace with actual local model execution."
        }
    
    def _execute_with_evolved_model(self, task_description: str, **kwargs) -> Dict[str, Any]:
        """Execute task with evolved model (mock implementation)"""
        # Mock evolved model execution
        time.sleep(1.5)  # Simulate processing time
        
        return {
            "status": "success", 
            "model": "evolved_model",
            "response": "Task completed with evolved model (mock execution)",
            "note": "This is a mock implementation. Replace with actual evolved model execution."
        }
    
    def _record_performance(self, model_name: str, task_description: str, result: Dict[str, Any], execution_time: float):
        """Record performance metrics for model selection learning"""
        timestamp = datetime.now()
        
        if model_name not in self.performance_history:
            self.performance_history[model_name] = []
        
        performance_record = {
            "timestamp": timestamp.isoformat(),
            "task_description": task_description,
            "execution_time": execution_time,
            "status": result.get("status", "unknown"),
            "success": result.get("status") == "success",
            "task_complexity": self.analyze_task(task_description).complexity.value
        }
        
        self.performance_history[model_name].append(performance_record)
        
        # Keep only last 100 records per model
        if len(self.performance_history[model_name]) > 100:
            self.performance_history[model_name] = self.performance_history[model_name][-100:]
    
    def _load_performance_history(self):
        """Load performance history from disk"""
        try:
            history_file = f"{self.performance_dir}/model_performance_history.json"
            if os.path.exists(history_file):
                with open(history_file, 'r', encoding='utf-8') as f:
                    self.performance_history = json.load(f)
        except Exception as e:
            print(f"Warning: Could not load performance history: {e}")
    
    def get_model_performance_summary(self) -> Dict[str, Dict[str, Any]]:
        """Get performance summary for all models"""
        summaries = {}
        
        for model_name, records in self.performance_history.items():
            if not records:
                continue
                
            total_executions = len(records)
            successful_executions = sum(1 for r in records if r.get("success", False))
            success_rate = successful_executions / total_executions if total_executions > 0 else 0
            
            avg_execution_time = sum(r.get("execution_time", 0) for r in records) / total_executions if total_executions > 0 else 0
            
            # Get complexity distribution
            complexity_counts = {}
            for record in records:
                complexity = record.get("task_complexity", "unknown")
                complexity_counts[complexity] = complexity_counts.get(complexity, 0) + 1
            
            summaries[model_name] = {
                "total_executions": total_executions,
                "successful_executions": successful_executions,
                "success_rate": success_rate,
                "avg_execution_time": avg_execution_time,
                "complexity_distribution": complexity_counts
            }
        
        return summaries
    
    def export_performance_report(self, output_file: str = None) -> str:
        """Export performance report to file"""
        if not output_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"{self.performance_dir}/performance_report_{timestamp}.json"
        
        report_data = {
            "generated_at": datetime.now().isoformat(),
            "model_summaries": self.get_model_performance_summary(),
            "performance_history": self.performance_history
        }
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2, ensure_ascii=False)
            return output_file
        except Exception as e:
            print(f"Error exporting performance report: {e}")
            return ""

# Convenience functions
def create_model_manager(config_path: str = "configs/azure_config.yaml") -> DynamicModelManager:
    """Create and return a dynamic model manager instance"""
    return DynamicModelManager(config_path)

if __name__ == "__main__":
    # Example usage
    manager = create_model_manager()
    
    print("🤖 OpenMLAIDS Dynamic Model Manager")
    print("=" * 50)
    
    # Test task analysis
    test_tasks = [
        "What is machine learning?",
        "Analyze this sales dataset and create predictive models",
        "Research and develop a comprehensive end-to-end AI system for fraud detection"
    ]
    
    for task in test_tasks:
        profile = manager.analyze_task(task)
        print(f"\nTask: {task}")
        print(f"Complexity: {profile.complexity.value}")
        print(f"Domain: {profile.domain}")
        print(f"Requires coding: {profile.requires_coding}")
        print(f"Requires reasoning: {profile.requires_reasoning}")
        
        # Test model selection
        selected_model, confidence = manager.select_optimal_model(profile)
        print(f"Selected model: {selected_model} (confidence: {confidence:.2f})")
