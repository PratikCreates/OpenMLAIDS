# -*- coding: utf-8 -*-
"""
OpenMLAIDS + Oumi Performance Tracking System
Comprehensive performance monitoring and analytics for the self-evolving AI agent
"""

import os
import json
import time
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import sqlite3

# Import our systems
from .judge_manager import judge_manager
from .fine_tuning_pipeline import create_evolution_pipeline
from .model_manager import create_model_manager
from .continuous_improvement_loop import create_improvement_loop

@dataclass
class PerformanceSnapshot:
    """Snapshot of system performance at a point in time"""
    timestamp: str
    total_interactions: int
    success_rate: float
    average_quality_score: float
    model_performance: Dict[str, float]
    evolution_count: int
    training_data_size: int
    system_health: str  # "healthy", "degraded", "critical"
    trends: Dict[str, str]  # "improving", "stable", "declining"

@dataclass
class MetricSeries:
    """Time series of performance metrics"""
    timestamps: List[str]
    values: List[float]
    metric_name: str
    unit: str
    trend: str

class PerformanceTracker:
    """
    Comprehensive performance tracking and analytics system
    """
    
    def __init__(self, config_path: str = "configs/azure_config.yaml"):
        """Initialize the performance tracker"""
        self.config_path = config_path
        self.performance_dir = "models/performance"
        self.tracking_db = f"{self.performance_dir}/performance_tracking.db"
        self.metrics_file = f"{self.performance_dir}/performance_metrics.json"
        self.reports_dir = f"{self.performance_dir}/reports"
        self.charts_dir = f"{self.performance_dir}/charts"
        
        # Create directories
        Path(self.performance_dir).mkdir(parents=True, exist_ok=True)
        Path(self.reports_dir).mkdir(parents=True, exist_ok=True)
        Path(self.charts_dir).mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self._init_database()
        
        # Initialize components
        self.model_manager = create_model_manager(config_path)
        self.improvement_loop = create_improvement_loop(config_path)
        
        print("📊 Performance Tracker initialized")
        print(f"   Database: {self.tracking_db}")
        print(f"   Reports: {self.reports_dir}")
        print(f"   Charts: {self.charts_dir}")
    
    def _init_database(self):
        """Initialize SQLite database for performance tracking"""
        try:
            conn = sqlite3.connect(self.tracking_db)
            cursor = conn.cursor()
            
            # Create tables
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS performance_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    total_interactions INTEGER NOT NULL,
                    success_rate REAL NOT NULL,
                    average_quality_score REAL NOT NULL,
                    model_performance TEXT NOT NULL,
                    evolution_count INTEGER NOT NULL,
                    training_data_size INTEGER NOT NULL,
                    system_health TEXT NOT NULL,
                    trends TEXT NOT NULL
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS metric_series (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    metric_name TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    value REAL NOT NULL,
                    metadata TEXT
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS evolution_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    evolution_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    performance_score REAL,
                    improvement_percentage REAL,
                    details TEXT
                )
            ''')
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            print(f"Warning: Could not initialize database: {e}")
    
    def record_metric(self, metric_name: str, value: float, metadata: Dict[str, Any] = None):
        """
        Record a performance metric
        
        Args:
            metric_name: Name of the metric
            value: Metric value
            metadata: Additional metadata
        """
        try:
            conn = sqlite3.connect(self.tracking_db)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO metric_series (metric_name, timestamp, value, metadata)
                VALUES (?, ?, ?, ?)
            ''', (
                metric_name,
                datetime.now().isoformat(),
                value,
                json.dumps(metadata) if metadata else None
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            print(f"Warning: Could not record metric {metric_name}: {e}")
    
    def take_snapshot(self) -> PerformanceSnapshot:
        """
        Take a performance snapshot
        
        Returns:
            PerformanceSnapshot object
        """
        try:
            # Get current metrics from improvement loop
            improvement_status = self.improvement_loop.get_improvement_status()
            metrics = improvement_status['metrics']
            
            # Get model performance
            model_summaries = self.model_manager.get_model_performance_summary()
            model_performance = {}
            for model, summary in model_summaries.items():
                if 'error' not in summary:
                    model_performance[model] = summary.get('success_rate', 0.0)
            
            # Determine system health
            system_health = self._assess_system_health(metrics, model_performance)
            
            # Determine trends
            trends = self._calculate_trends()
            
            # Create snapshot
            snapshot = PerformanceSnapshot(
                timestamp=datetime.now().isoformat(),
                total_interactions=metrics['total_interactions'],
                success_rate=metrics['success_rate'],
                average_quality_score=metrics['average_quality_score'],
                model_performance=model_performance,
                evolution_count=metrics['evolution_count'],
                training_data_size=metrics['training_data_samples'],
                system_health=system_health,
                trends=trends
            )
            
            # Save snapshot to database
            self._save_snapshot(snapshot)
            
            return snapshot
            
        except Exception as e:
            print(f"Warning: Could not take snapshot: {e}")
            # Return default snapshot
            return PerformanceSnapshot(
                timestamp=datetime.now().isoformat(),
                total_interactions=0,
                success_rate=0.0,
                average_quality_score=0.0,
                model_performance={},
                evolution_count=0,
                training_data_size=0,
                system_health="unknown",
                trends={}
            )
    
    def _assess_system_health(self, metrics: Dict, model_performance: Dict[str, float]) -> str:
        """Assess overall system health"""
        try:
            # Check success rate
            if metrics['success_rate'] < 0.5:
                return "critical"
            elif metrics['success_rate'] < 0.7:
                return "degraded"
            
            # Check model performance
            if model_performance:
                avg_model_performance = sum(model_performance.values()) / len(model_performance)
                if avg_model_performance < 0.6:
                    return "critical"
                elif avg_model_performance < 0.8:
                    return "degraded"
            
            return "healthy"
            
        except Exception:
            return "unknown"
    
    def _calculate_trends(self) -> Dict[str, str]:
        """Calculate performance trends"""
        trends = {}
        
        try:
            # Get recent metrics (last 24 hours)
            recent_metrics = self._get_recent_metrics(hours=24)
            
            if len(recent_metrics) < 2:
                return {"status": "insufficient_data"}
            
            # Calculate trends for key metrics
            success_rates = [m.get('success_rate', 0) for m in recent_metrics]
            quality_scores = [m.get('average_quality_score', 0) for m in recent_metrics]
            
            # Success rate trend
            if len(success_rates) >= 5:
                early_avg = np.mean(success_rates[:len(success_rates)//2])
                late_avg = np.mean(success_rates[len(success_rates)//2:])
                
                if late_avg > early_avg * 1.05:
                    trends['success_rate'] = "improving"
                elif late_avg < early_avg * 0.95:
                    trends['success_rate'] = "declining"
                else:
                    trends['success_rate'] = "stable"
            
            # Quality score trend
            if len(quality_scores) >= 5:
                early_avg = np.mean(quality_scores[:len(quality_scores)//2])
                late_avg = np.mean(quality_scores[len(quality_scores)//2:])
                
                if late_avg > early_avg * 1.05:
                    trends['quality_score'] = "improving"
                elif late_avg < early_avg * 0.95:
                    trends['quality_score'] = "declining"
                else:
                    trends['quality_score'] = "stable"
            
        except Exception as e:
            print(f"Warning: Could not calculate trends: {e}")
        
        return trends
    
    def _get_recent_metrics(self, hours: int = 24) -> List[Dict]:
        """Get recent performance metrics from database"""
        try:
            conn = sqlite3.connect(self.tracking_db)
            cursor = conn.cursor()
            
            cutoff_time = (datetime.now() - timedelta(hours=hours)).isoformat()
            
            cursor.execute('''
                SELECT * FROM performance_snapshots 
                WHERE timestamp > ? 
                ORDER BY timestamp DESC
            ''', (cutoff_time,))
            
            rows = cursor.fetchall()
            conn.close()
            
            # Convert to dictionaries
            metrics = []
            for row in rows:
                metrics.append({
                    'timestamp': row[1],
                    'total_interactions': row[2],
                    'success_rate': row[3],
                    'average_quality_score': row[4],
                    'model_performance': json.loads(row[5]),
                    'evolution_count': row[6],
                    'training_data_size': row[7],
                    'system_health': row[8],
                    'trends': json.loads(row[9])
                })
            
            return metrics
            
        except Exception as e:
            print(f"Warning: Could not get recent metrics: {e}")
            return []
    
    def _save_snapshot(self, snapshot: PerformanceSnapshot):
        """Save snapshot to database"""
        try:
            conn = sqlite3.connect(self.tracking_db)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO performance_snapshots (
                    timestamp, total_interactions, success_rate, average_quality_score,
                    model_performance, evolution_count, training_data_size,
                    system_health, trends
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                snapshot.timestamp,
                snapshot.total_interactions,
                snapshot.success_rate,
                snapshot.average_quality_score,
                json.dumps(snapshot.model_performance),
                snapshot.evolution_count,
                snapshot.training_data_size,
                snapshot.system_health,
                json.dumps(snapshot.trends)
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            print(f"Warning: Could not save snapshot: {e}")
    
    def generate_performance_report(self, output_file: str = None) -> str:
        """
        Generate comprehensive performance report
        
        Args:
            output_file: Output file path (optional)
            
        Returns:
            Path to generated report
        """
        if not output_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"{self.reports_dir}/performance_report_{timestamp}.md"
        
        try:
            # Get latest snapshot
            latest_snapshot = self.take_snapshot()
            
            # Get historical data
            recent_snapshots = self._get_recent_metrics(hours=168)  # Last week
            
            # Generate report content
            report_content = self._generate_report_content(latest_snapshot, recent_snapshots)
            
            # Save report
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report_content)
            
            print(f"📊 Performance report generated: {output_file}")
            return output_file
            
        except Exception as e:
            print(f"Warning: Could not generate report: {e}")
            return ""
    
    def _generate_report_content(self, latest_snapshot: PerformanceSnapshot, recent_snapshots: List[Dict]) -> str:
        """Generate report content"""
        content = f"""# OpenMLAIDS Performance Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary

### System Health: {latest_snapshot.system_health.upper()}

- **Total Interactions**: {latest_snapshot.total_interactions:,}
- **Success Rate**: {latest_snapshot.success_rate:.1%}
- **Average Quality Score**: {latest_snapshot.average_quality_score:.1f}/10
- **Evolution Count**: {latest_snapshot.evolution_count}
- **Training Data Size**: {latest_snapshot.training_data_size:,} samples

### Key Trends
"""
        
        for trend_name, trend_value in latest_snapshot.trends.items():
            trend_emoji = "📈" if trend_value == "improving" else "📉" if trend_value == "declining" else "➡️"
            content += f"- **{trend_name.replace('_', ' ').title()}**: {trend_emoji} {trend_value}\n"
        
        content += "\n## Model Performance\n\n"
        
        for model, performance in latest_snapshot.model_performance.items():
            status_emoji = "🟢" if performance > 0.8 else "🟡" if performance > 0.6 else "🔴"
            content += f"- **{model}**: {status_emoji} {performance:.1%}\n"
        
        content += "\n## Recent Activity (Last 7 Days)\n\n"
        
        if recent_snapshots:
            content += f"- **Snapshots captured**: {len(recent_snapshots)}\n"
            content += f"- **Peak success rate**: {max(s['success_rate'] for s in recent_snapshots):.1%}\n"
            content += f"- **Lowest success rate**: {min(s['success_rate'] for s in recent_snapshots):.1%}\n"
            content += f"- **Evolutions triggered**: {max(s['evolution_count'] for s in recent_snapshots) - min(s['evolution_count'] for s in recent_snapshots)}\n"
        else:
            content += "- No recent activity data available\n"
        
        content += "\n## Recommendations\n\n"
        
        # Add recommendations based on system health
        if latest_snapshot.system_health == "critical":
            content += "🔴 **CRITICAL**: System performance is below acceptable levels. Immediate intervention recommended.\n\n"
        elif latest_snapshot.system_health == "degraded":
            content += "🟡 **WARNING**: System performance is below optimal levels. Monitor closely and consider evolution.\n\n"
        else:
            content += "🟢 **HEALTHY**: System performance is within acceptable parameters.\n\n"
        
        content += "### Next Steps\n"
        content += "- Continue monitoring system performance\n"
        content += "- Review evolution criteria and adjust if needed\n"
        content += "- Consider manual evolution if trends remain negative\n"
        content += "- Analyze model switching patterns for optimization\n"
        
        return content
    
    def create_performance_charts(self, output_dir: str = None) -> List[str]:
        """
        Create performance visualization charts
        
        Args:
            output_dir: Output directory for charts
            
        Returns:
            List of generated chart file paths
        """
        if not output_dir:
            output_dir = self.charts_dir
        
        chart_files = []
        
        try:
            # Success rate over time
            chart_file = f"{output_dir}/success_rate_trend.png"
            self._create_success_rate_chart(chart_file)
            chart_files.append(chart_file)
            
            # Model performance comparison
            chart_file = f"{output_dir}/model_performance.png"
            self._create_model_performance_chart(chart_file)
            chart_files.append(chart_file)
            
            # Evolution timeline
            chart_file = f"{output_dir}/evolution_timeline.png"
            self._create_evolution_chart(chart_file)
            chart_files.append(chart_file)
            
            print(f"📈 Generated {len(chart_files)} performance charts")
            return chart_files
            
        except Exception as e:
            print(f"Warning: Could not create charts: {e}")
            return chart_files
    
    def _create_success_rate_chart(self, output_file: str):
        """Create success rate trend chart"""
        try:
            # Get data
            snapshots = self._get_recent_metrics(hours=168)  # Last week
            
            if not snapshots:
                return
            
            # Prepare data
            timestamps = [datetime.fromisoformat(s['timestamp']) for s in snapshots]
            success_rates = [s['success_rate'] for s in snapshots]
            
            # Create chart
            plt.figure(figsize=(12, 6))
            plt.plot(timestamps, success_rates, marker='o', linewidth=2, markersize=4)
            plt.title('Success Rate Trend (Last 7 Days)')
            plt.xlabel('Time')
            plt.ylabel('Success Rate')
            plt.ylim(0, 1)
            plt.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"Warning: Could not create success rate chart: {e}")
    
    def _create_model_performance_chart(self, output_file: str):
        """Create model performance comparison chart"""
        try:
            # Get latest snapshot for model performance data
            latest_snapshot = self.take_snapshot()
            model_performance = latest_snapshot.model_performance
            
            if not model_performance:
                return
            
            # Prepare data
            models = list(model_performance.keys())
            performances = [model_performance[model] for model in models]
            
            # Create chart
            plt.figure(figsize=(10, 6))
            bars = plt.bar(models, performances, color=['#2ea043', '#1f6feb', '#da3633', '#bf616a'])
            plt.title('Model Performance Comparison')
            plt.xlabel('Models')
            plt.ylabel('Success Rate')
            plt.ylim(0, 1)
            plt.grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars
            for bar, perf in zip(bars, performances):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{perf:.1%}', ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"Warning: Could not create model performance chart: {e}")
    
    def _create_evolution_chart(self, output_file: str):
        """Create evolution timeline chart"""
        try:
            # Get recent metrics for evolution data
            recent_snapshots = self._get_recent_metrics(hours=168)  # Last week
            
            if not recent_snapshots:
                return
            
            # Prepare data
            timestamps = [datetime.fromisoformat(s['timestamp']) for s in recent_snapshots]
            evolution_counts = [s['evolution_count'] for s in recent_snapshots]
            
            # Create chart
            plt.figure(figsize=(12, 6))
            plt.plot(timestamps, evolution_counts, marker='s', linewidth=2, markersize=4, color='#6e40aa')
            plt.title('Evolution Timeline')
            plt.xlabel('Time')
            plt.ylabel('Evolution Count')
            plt.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"Warning: Could not create evolution chart: {e}")
    
    def get_performance_insights(self) -> Dict[str, Any]:
        """Get performance insights and recommendations"""
        try:
            latest_snapshot = self.take_snapshot()
            recent_snapshots = self._get_recent_metrics(hours=168)
            
            insights = {
                "system_health": latest_snapshot.system_health,
                "performance_trends": latest_snapshot.trends,
                "model_performance": latest_snapshot.model_performance,
                "recent_activity": {
                    "total_snapshots": len(recent_snapshots),
                    "peak_success_rate": max((s['success_rate'] for s in recent_snapshots), default=0),
                    "lowest_success_rate": min((s['success_rate'] for s in recent_snapshots), default=0)
                }
            }
            
            # Add recommendations
            recommendations = []
            if latest_snapshot.system_health == "critical":
                recommendations.append("Immediate system intervention required")
                recommendations.append("Review recent failures and error logs")
            elif latest_snapshot.system_health == "degraded":
                recommendations.append("Monitor system performance closely")
                recommendations.append("Consider triggering manual evolution")
            
            if "declining" in latest_snapshot.trends.values():
                recommendations.append("Investigate declining performance trends")
                recommendations.append("Review recent model changes or updates")
            
            insights["recommendations"] = recommendations
            return insights
            
        except Exception as e:
            print(f"Warning: Could not generate performance insights: {e}")
            return {"error": str(e)}

# Convenience functions
def create_performance_tracker(config_path: str = "configs/azure_config.yaml") -> PerformanceTracker:
    """Create and return a performance tracker instance"""
    return PerformanceTracker(config_path)

if __name__ == "__main__":
    # Example usage
    tracker = create_performance_tracker()
    
    print("🤖 OpenMLAIDS Performance Tracker")
    print("=" * 50)
    
    # Take a snapshot
    snapshot = tracker.take_snapshot()
    print(f"Latest snapshot: {snapshot.timestamp}")
    print(f"System health: {snapshot.system_health}")
    print(f"Success rate: {snapshot.success_rate:.1%}")
    
    # Generate report
    report_file = tracker.generate_performance_report()
    if report_file:
        print(f"Performance report saved to: {report_file}")
    
    # Get insights
    insights = tracker.get_performance_insights()
    print(f"\nPerformance Insights:")
    print(f"Health: {insights.get('system_health', 'unknown')}")
    print(f"Trends: {insights.get('performance_trends', {})}")
