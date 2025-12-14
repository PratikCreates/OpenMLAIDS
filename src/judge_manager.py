"""
Judge Manager for OpenMLAIDS Self-Evolving AI Agent
Handles loading and managing Oumi Simple Judge configurations
"""

import os
import yaml
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import asyncio
from oumi.judges.simple_judge import SimpleJudge
from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv

load_dotenv()

@dataclass
class JudgeResult:
    """Structure for judge evaluation results"""
    judge_name: str
    score: float
    explanation: str
    detailed_feedback: Dict[str, Any]
    timestamp: str

class JudgeManager:
    """Manages multiple Oumi Simple Judge configurations for agent evaluation"""
    
    def __init__(self, config_dir: str = "configs/judge_configs"):
        self.config_dir = Path(config_dir)
        self.judges: Dict[str, SimpleJudge] = {}
        self.azure_llm = None
        self._load_judges()
        self._setup_azure_llm()
    
    def _load_judges(self):
        """Load all judge configurations from the config directory"""
        if not self.config_dir.exists():
            print(f"Warning: Judge config directory {self.config_dir} does not exist")
            return
            
        for config_file in self.config_dir.glob("*.yaml"):
            try:
                judge_name = config_file.stem
                print(f"Loading judge: {judge_name}")
                self.judges[judge_name] = SimpleJudge(str(config_file))
            except Exception as e:
                print(f"Error loading judge {judge_name}: {e}")
    
    def _setup_azure_llm(self):
        """Setup Azure OpenAI LLM for judges"""
        try:
            self.azure_llm = AzureChatOpenAI(
                azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
                api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                extra_body={
                    "reasoning_effort": "medium", 
                    "verbosity": "medium",
                    "max_completion_tokens": 4096
                }
            )
            print("Azure OpenAI LLM configured successfully for judges")
        except Exception as e:
            print(f"Warning: Could not setup Azure LLM: {e}")
    
    def evaluate_agent_response(self, 
                          user_request: str, 
                          agent_response: str, 
                          response_type: str = "general") -> Dict[str, JudgeResult]:
        """
        Evaluate agent response using all available judges
        
        Args:
            user_request: The original user request
            agent_response: The agent's response to evaluate
            response_type: Type of response ("general", "code", "data_science", "problem_solving")
            
        Returns:
            Dictionary of judge_name -> JudgeResult
        """
        results = {}
        
        # Select relevant judges based on response type
        relevant_judges = self._select_judges(response_type)
        
        for judge_name in relevant_judges:
            if judge_name not in self.judges:
                continue
                
            try:
                # Prepare dataset for this judge
                dataset = [{
                    "user_request": user_request,
                    f"{response_type}_response": agent_response
                }]
                
                # Run evaluation
                outputs = self.judges[judge_name].judge(dataset)
                
                if outputs:
                    output = outputs[0]
                    field_values = output.field_values
                    
                    # Extract score and explanation
                    score = self._extract_score(field_values)
                    explanation = field_values.get("explanation", "No explanation provided")
                    
                    results[judge_name] = JudgeResult(
                        judge_name=judge_name,
                        score=score,
                        explanation=explanation,
                        detailed_feedback=field_values,
                        timestamp=str(asyncio.get_event_loop().time())
                    )
                    
            except Exception as e:
                print(f"Error evaluating with {judge_name}: {e}")
                results[judge_name] = JudgeResult(
                    judge_name=judge_name,
                    score=0.0,
                    explanation=f"Error during evaluation: {e}",
                    detailed_feedback={"error": str(e)},
                    timestamp=str(asyncio.get_event_loop().time())
                )
        
        return results
    
    def _select_judges(self, response_type: str) -> List[str]:
        """Select appropriate judges based on response type"""
        judge_mapping = {
            "general": ["agent_evaluation_judge", "problem_solving_judge"],
            "code": ["code_quality_judge", "agent_evaluation_judge"],
            "data_science": ["data_science_judge", "agent_evaluation_judge"],
            "problem_solving": ["problem_solving_judge", "agent_evaluation_judge"]
        }
        return judge_mapping.get(response_type, ["agent_evaluation_judge"])
    
    def _extract_score(self, field_values: Dict[str, Any]) -> float:
        """Extract numeric score from judge output"""
        # Try different possible score field names
        score_keys = ["score", "judgment", "overall_score", "total_score"]
        
        for key in score_keys:
            if key in field_values:
                try:
                    return float(field_values[key])
                except (ValueError, TypeError):
                    continue
        
        # If no explicit score, try to extract from explanation or other fields
        for value in field_values.values():
            if isinstance(value, (int, float)):
                return float(value)
            elif isinstance(value, str) and value.replace(".", "").isdigit():
                return float(value)
        
        return 0.0  # Default score if none found
    
    def get_judge_list(self) -> List[str]:
        """Get list of available judges"""
        return list(self.judges.keys())
    
    def evaluate_with_specific_judge(self, 
                                judge_name: str, 
                                user_request: str, 
                                agent_response: str, 
                                response_type: str = "general") -> Optional[JudgeResult]:
        """Evaluate using a specific judge"""
        if judge_name not in self.judges:
            return None
            
        results = self.evaluate_agent_response(user_request, agent_response, response_type)
        return results.get(judge_name)
    
    def get_performance_summary(self, results: Dict[str, JudgeResult]) -> Dict[str, Any]:
        """Get performance summary from judge results"""
        if not results:
            return {"error": "No results to summarize"}
        
        scores = [result.score for result in results.values()]
        avg_score = sum(scores) / len(scores) if scores else 0.0
        
        return {
            "average_score": avg_score,
            "num_judges": len(results),
            "judge_scores": {name: result.score for name, result in results.items()},
            "best_judge": max(results.items(), key=lambda x: x[1].score)[0] if results else None,
            "worst_judge": min(results.items(), key=lambda x: x[1].score)[0] if results else None,
            "needs_improvement": avg_score < 7.0,  # Threshold for requiring improvement
            "feedback_summary": [result.explanation for result in results.values()]
        }

# Global judge manager instance
judge_manager = JudgeManager()
