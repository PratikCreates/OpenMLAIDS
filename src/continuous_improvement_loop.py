# -*- coding: utf-8 -*-
"""
OpenMLAIDS + Oumi Continuous Improvement Loop
Orchestrates continuous learning and evolution of the AI agent system
"""

import os
import json
import time
import yaml
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import schedule
import pandas as pd

# Import our systems
from .judge_manager import judge_manager
from .fine_tuning_pipeline import SelfEvolutionPipeline, create_evolution_pipeline
from .model_manager import DynamicModelManager, create_model_manager

@dataclass
class ImprovementMetrics:
    """Metrics for tracking improvement over time"""
    timestamp: str
    total_interactions: int
    successful_interactions: int
    success_rate: float
    average_quality_score: float
    evolution_count: int
    model_performance: Dict[str, float]
    training_data_samples: int
    last_evolution_time: Optional[str]

class ContinuousImprovementLoop:
    """
    Continuous improvement loop that orchestrates self-evolution
    """
    
    def __init__(self, config_path: str = "configs/azure_config.yaml"):
        """Initialize the continuous improvement loop"""
        self.config_path = config_path
        self.metrics_file = "models/performance/improvement_metrics.json"
        self.interaction_log_file = "models/performance/interaction_log.jsonl"
        self.improvement_config_file = "configs/improvement_config.yaml"
        
        # Create performance directory
        Path("models/performance").mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.judge_manager = judge_manager
        self.evolution_pipeline = create_evolution_pipeline(config_path)
        self.model_manager = create_model_manager(config_path)
        
        # Improvement state
        self.is_running = False
        self.improvement_thread = None
        self.metrics = self._load_metrics()
        self.interaction_buffer = []
        
        # Load or create improvement config
        self.improvement_config = self._load_improvement_config()
        
        print("🔄 Continuous Improvement Loop initialized")
        print(f"   Metrics file: {self.metrics_file}")
        print(f"   Evolution threshold: {self.improvement_config['evolution_threshold']}")
        print(f"   Check interval: {self.improvement_config['check_interval_hours']} hours")
    
    def _load_improvement_config(self) -> Dict:
        """Load improvement configuration"""
        default_config = {
            "evolution_threshold": 50,  # Minimum high-quality interactions before evolution
            "check_interval_hours": 24,  # Check for evolution every 24 hours
            "min_quality_score": 7.0,  # Minimum score for "high-quality" interaction
            "performance_decline_threshold": -0.1,  # Trigger evolution if performance drops by 10%
            "max_buffer_size": 1000,  # Maximum interactions to buffer before processing
            "auto_evolution": True,  # Automatically trigger evolution when criteria met
            "performance_tracking": {
                "track_model_switching": True,
                "track_quality_scores": True,
                "track_execution_times": True
            }
        }
        
        try:
            if os.path.exists(self.improvement_config_file):
                with open(self.improvement_config_file, 'r') as f:
                    config = yaml.safe_load(f)
                    # Merge with defaults
                    for key, value in default_config.items():
                        if key not in config:
                            config[key] = value
                    return config
        except Exception as e:
            print(f"Warning: Could not load improvement config: {e}")
        
        # Save default config
        with open(self.improvement_config_file, 'w') as f:
            yaml.dump(default_config, f, default_flow_style=False)
        
        return default_config
    
    def _load_metrics(self) -> ImprovementMetrics:
        """Load existing improvement metrics"""
        try:
            if os.path.exists(self.metrics_file):
                with open(self.metrics_file, 'r') as f:
                    data = json.load(f)
                    return ImprovementMetrics(**data)
        except Exception as e:
            print(f"Warning: Could not load metrics: {e}")
        
        # Return default metrics
        return ImprovementMetrics(
            timestamp=datetime.now().isoformat(),
            total_interactions=0,
            successful_interactions=0,
            success_rate=0.0,
            average_quality_score=0.0,
            evolution_count=0,
            model_performance={},
            training_data_samples=0,
            last_evolution_time=None
        )
    
    def _save_metrics(self):
        """Save current improvement metrics"""
        try:
            with open(self.metrics_file, 'w') as f:
                json.dump(asdict(self.metrics), f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save metrics: {e}")
    
    def log_interaction(self, interaction_data: Dict[str, Any]):
        """
        Log an agent interaction for improvement tracking
        
        Args:
            interaction_data: Dictionary containing interaction details
        """
        interaction_data['timestamp'] = datetime.now().isoformat()
        self.interaction_buffer.append(interaction_data)
        
        # Update metrics
        self.metrics.total_interactions += 1
        if interaction_data.get('success', False):
            self.metrics.successful_interactions += 1
        
        # Calculate running averages
        self.metrics.success_rate = (
            self.metrics.successful_interactions / self.metrics.total_interactions
        )
        
        # Save interaction to log file
        try:
            with open(self.interaction_log_file, 'a') as f:
                f.write(json.dumps(interaction_data) + '\n')
        except Exception as e:
            print(f"Warning: Could not log interaction: {e}")
        
        # Process buffer if full
        if len(self.interaction_buffer) >= self.improvement_config['max_buffer_size']:
            self._process_interaction_buffer()
    
    def _process_interaction_buffer(self):
        """Process buffered interactions for training data generation"""
        if not self.interaction_buffer:
            return
        
        print(f"📊 Processing {len(self.interaction_buffer)} buffered interactions...")
        
        # Generate training data from high-quality interactions
        high_quality_count = 0
        for interaction in self.interaction_buffer:
            if (interaction.get('quality_score', 0) >= 
                self.improvement_config['min_quality_score']):
                
                # Generate training data
                try:
                    training_example = {
                        "instruction": interaction.get('user_prompt', ''),
                        "response": interaction.get('agent_response', ''),
                        "evaluation": {
                            "score": interaction.get('quality_score', 0),
                            "feedback": interaction.get('feedback', ''),
                            "judge_type": interaction.get('judge_type', 'general')
                        },
                        "metadata": {
                            "timestamp": interaction['timestamp'],
                            "model": interaction.get('model_used', 'unknown'),
                            "framework": "OpenMLAIDS-v2.0"
                        }
                    }
                    
                    # Save to training data file
                    training_file = "data/training/training_data.jsonl"
                    os.makedirs(os.path.dirname(training_file), exist_ok=True)
                    with open(training_file, 'a') as f:
                        f.write(json.dumps(training_example) + '\n')
                    
                    high_quality_count += 1
                    
                except Exception as e:
                    print(f"Warning: Could not process interaction for training: {e}")
        
        self.metrics.training_data_samples += high_quality_count
        self.interaction_buffer.clear()
        
        print(f"✅ Generated {high_quality_count} training examples")
    
    def _check_evolution_criteria(self) -> Tuple[bool, str]:
        """
        Check if evolution criteria are met
        
        Returns:
            Tuple of (should_evolve, reason)
        """
        # Check if we have enough training data
        if self.metrics.training_data_samples < self.improvement_config['evolution_threshold']:
            return False, f"Need {self.improvement_config['evolution_threshold'] - self.metrics.training_data_samples} more training samples"
        
        # Check if enough time has passed since last evolution
        if self.metrics.last_evolution_time:
            last_evolution = datetime.fromisoformat(self.metrics.last_evolution_time)
            time_since = datetime.now() - last_evolution
            
            if time_since.total_seconds() < (self.improvement_config['check_interval_hours'] * 3600):
                hours_remaining = (
                    self.improvement_config['check_interval_hours'] - 
                    time_since.total_seconds() / 3600
                )
                return False, f"Wait {hours_remaining:.1f} more hours before next evolution"
        
        # Check performance trends
        if self.metrics.success_rate < 0.7:  # Success rate below 70%
            return True, f"Low success rate ({self.metrics.success_rate:.1%}) suggests need for improvement"
        
        # Check if performance has been declining
        recent_performance = self._get_recent_performance(24)  # Last 24 hours
        if recent_performance and len(recent_performance) > 10:
            avg_recent = sum(recent_performance) / len(recent_performance)
            if avg_recent < self.metrics.average_quality_score * 0.9:  # 10% decline
                return True, f"Performance declining: {avg_recent:.1f} vs {self.metrics.average_quality_score:.1f}"
        
        return True, "Evolution criteria met"
    
    def _get_recent_performance(self, hours: int) -> List[float]:
        """Get performance scores from recent interactions"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            recent_scores = []
            
            if os.path.exists(self.interaction_log_file):
                with open(self.interaction_log_file, 'r') as f:
                    for line in f:
                        if line.strip():
                            interaction = json.loads(line)
                            timestamp = datetime.fromisoformat(interaction['timestamp'])
                            if timestamp > cutoff_time:
                                score = interaction.get('quality_score', 0)
                                if score > 0:
                                    recent_scores.append(score)
            
            return recent_scores
        except Exception as e:
            print(f"Warning: Could not get recent performance: {e}")
            return []
    
    def _update_model_performance(self):
        """Update model performance metrics"""
        try:
            model_summaries = self.model_manager.get_model_performance_summary()
            
            for model_name, summary in model_summaries.items():
                if 'error' not in summary:
                    self.metrics.model_performance[model_name] = summary.get('success_rate', 0.0)
            
            # Calculate overall average quality score
            all_scores = []
            for score in self.metrics.model_performance.values():
                all_scores.append(score)
            
            if all_scores:
                self.metrics.average_quality_score = sum(all_scores) / len(all_scores)
            
        except Exception as e:
            print(f"Warning: Could not update model performance: {e}")
    
    def run_evolution_cycle(self) -> Dict[str, Any]:
        """
        Run a complete evolution cycle
        
        Returns:
            Evolution results
        """
        print("\n🔄 Starting evolution cycle...")
        
        # Process any buffered interactions
        self._process_interaction_buffer()
        
        # Update performance metrics
        self._update_model_performance()
        
        # Check evolution criteria
        should_evolve, reason = self._check_evolution_criteria()
        
        if not should_evolve:
            print(f"⏳ Evolution not triggered: {reason}")
            return {"status": "skipped", "reason": reason}
        
        print(f"🚀 Evolution triggered: {reason}")
        
        # Run evolution
        try:
            evolution_results = self.evolution_pipeline.evolve_agent()
            
            if evolution_results.get("status") == "success":
                self.metrics.evolution_count += 1
                self.metrics.last_evolution_time = datetime.now().isoformat()
                
                # Update evolved model performance in model manager
                if "evaluation_results" in evolution_results:
                    eval_score = evolution_results["evaluation_results"].get("performance_score", 0)
                    self.metrics.model_performance["evolved_model"] = eval_score
                
                print(f"✅ Evolution completed successfully!")
                return evolution_results
            else:
                print(f"❌ Evolution failed: {evolution_results.get('error', 'Unknown error')}")
                return evolution_results
                
        except Exception as e:
            error_msg = f"Evolution cycle failed: {str(e)}"
            print(f"❌ {error_msg}")
            return {"status": "error", "error": error_msg}
        
        finally:
            # Save updated metrics
            self._save_metrics()
    
    def start_continuous_improvement(self):
        """Start the continuous improvement loop"""
        if self.is_running:
            print("⚠️ Continuous improvement loop is already running")
            return
        
        self.is_running = True
        
        def improvement_worker():
            """Worker thread for continuous improvement"""
            print("🔄 Starting continuous improvement worker...")
            
            while self.is_running:
                try:
                    # Check for evolution every configured interval
                    schedule.every(self.improvement_config['check_interval_hours']).hours.do(
                        self.run_evolution_cycle
                    )
                    
                    # Run pending scheduled tasks
                    schedule.run_pending()
                    
                    # Sleep for 1 hour and check again
                    time.sleep(3600)
                    
                except Exception as e:
                    print(f"❌ Error in improvement worker: {e}")
                    time.sleep(300)  # Wait 5 minutes before retrying
        
        # Start worker thread
        self.improvement_thread = threading.Thread(target=improvement_worker, daemon=True)
        self.improvement_thread.start()
        
        print("✅ Continuous improvement loop started")
    
    def stop_continuous_improvement(self):
        """Stop the continuous improvement loop"""
        if not self.is_running:
            print("⚠️ Continuous improvement loop is not running")
            return
        
        self.is_running = False
        
        if self.improvement_thread and self.improvement_thread.is_alive():
            self.improvement_thread.join(timeout=10)
        
        print("🛑 Continuous improvement loop stopped")
    
    def get_improvement_status(self) -> Dict[str, Any]:
        """Get current improvement status"""
        return {
            "is_running": self.is_running,
            "metrics": asdict(self.metrics),
            "improvement_config": self.improvement_config,
            "buffer_size": len(self.interaction_buffer),
            "next_evolution_check": self._get_next_evolution_time(),
            "evolution_recommended": self._check_evolution_criteria()[0]
        }
    
    def _get_next_evolution_time(self) -> Optional[str]:
        """Get next scheduled evolution time"""
        if not self.metrics.last_evolution_time:
            return "No evolution yet scheduled"
        
        try:
            last_evolution = datetime.fromisoformat(self.metrics.last_evolution_time)
            next_check = last_evolution + timedelta(hours=self.improvement_config['check_interval_hours'])
            return next_check.isoformat()
        except Exception:
            return "Unknown"
    
    def force_evolution(self) -> Dict[str, Any]:
        """Force an evolution cycle (bypass criteria)"""
        print("⚡ Forcing evolution cycle...")
        return self.run_evolution_cycle()


# Convenience functions
def create_improvement_loop(config_path: str = "configs/azure_config.yaml") -> ContinuousImprovementLoop:
    """Create and return a continuous improvement loop instance"""
    return ContinuousImprovementLoop(config_path)

def start_improvement_loop(config_path: str = "configs/azure_config.yaml"):
    """Start the continuous improvement loop"""
    loop = create_improvement_loop(config_path)
    loop.start_continuous_improvement()
    return loop

def log_agent_interaction(interaction_data: Dict[str, Any], config_path: str = "configs/azure_config.yaml"):
    """Log an agent interaction for improvement tracking"""
    loop = create_improvement_loop(config_path)
    loop.log_interaction(interaction_data)


if __name__ == "__main__":
    # Example usage
    loop = ContinuousImprovementLoop()
    
    print("🤖 OpenMLAIDS Continuous Improvement Loop")
    print("=" * 50)
    
    # Show current status
    status = loop.get_improvement_status()
    print(f"Status: {'Running' if status['is_running'] else 'Stopped'}")
    print(f"Total interactions: {status['metrics']['total_interactions']}")
    print(f"Success rate: {status['metrics']['success_rate']:.1%}")
    print(f"Evolution count: {status['metrics']['evolution_count']}")
    print(f"Training samples: {status['metrics']['training_data_samples']}")
    
    # Test evolution check
    should_evolve, reason = loop._check_evolution_criteria()
    print(f"Evolution recommended: {should_evolve}")
    print(f"Reason: {reason}")
    
    # Start continuous improvement
    print("\n🚀 Starting continuous improvement loop...")
    loop.start_continuous_improvement()
    
    try:
        # Keep running for demonstration
        while True:
            time.sleep(10)
            status = loop.get_improvement_status()
            if status['evolution_recommended']:
                print("🔄 Evolution recommended, running cycle...")
                result = loop.run_evolution_cycle()
                print(f"Evolution result: {result.get('status', 'unknown')}")
                
    except KeyboardInterrupt:
        print("\n🛑 Stopping improvement loop...")
        loop.stop_continuous_improvement()
