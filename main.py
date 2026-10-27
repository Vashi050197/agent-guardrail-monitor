"""
Unified CLI Entrypoint for the Weak-to-Strong Agent Guardrail & Observability System.
Usage:
    python3 main.py --simulate     # Run benchmark simulation with benign & adversarial agents
    python3 main.py --report       # Print metrics report (ROC-AUC, TPR@FPR=0.01, Precision/Recall)
    python3 main.py --dashboard    # Generate and output dashboard.html path
    python3 main.py --list         # List all recorded sessions and statuses
"""

import sys
import argparse
from simulate import run_simulation
from analytics import PerformanceAnalytics
from visualizer import DashboardGenerator
from database import DataStore

def main():
    parser = argparse.ArgumentParser(
        description="Weak-to-Strong LLM Agent Guardrail & Observability System (ICLR 2026)"
    )
    parser.add_argument("--simulate", action="store_true", help="Run benchmark simulation")
    parser.add_argument("--report", action="store_true", help="Print metrics and evaluation summary")
    parser.add_argument("--dashboard", action="store_true", help="Generate interactive HTML dashboard")
    parser.add_argument("--list", action="store_true", help="List all recorded agent sessions")
    parser.add_argument("--serve", action="store_true", help="Start web server and API for deployment")
    parser.add_argument("--port", type=int, default=8080, help="Port to run web server on (default: 8080)")
    parser.add_argument("--db", default="agent_monitor.db", help="Path to SQLite database")

    args = parser.parse_args()

    # Default action if no flags provided: run simulation and generate dashboard
    if not (args.simulate or args.report or args.dashboard or args.list):
        print("No specific flag provided. Running full simulation and dashboard generation...")
        run_simulation()
        return

    if args.serve:
        from server import start_server
        start_server(args.port)
        return

    if args.simulate:
        run_simulation()

    if args.report:
        analytics = PerformanceAnalytics(args.db)
        metrics = analytics.compute_adversarial_metrics()
        perf = analytics.get_time_series_performance()
        print("\n" + "=" * 65)
        print("             WEAK-TO-STRONG MONITORING EVALUATION REPORT")
        print("=" * 65)
        print(f"Total Sessions Evaluated:    {metrics.get('total_sessions', 0)}")
        print(f"Total Steps Audited:         {perf.get('total_audited_steps', 0)}")
        print(f"Benchmark ROC-AUC:           {metrics.get('auc_roc', 'N/A')}")
        print(f"TPR at FPR=0.01:             {metrics.get('tpr_at_fpr_01', 'N/A')}")
        cm = metrics.get('confusion_matrix', {})
        print(f"Precision:                   {cm.get('precision', 'N/A')}")
        print(f"Recall:                      {cm.get('recall', 'N/A')}")
        print(f"F1 Score:                    {cm.get('f1_score', 'N/A')}")
        print(f"Compute Efficiency:          {metrics.get('token_efficiency_gain')}")
        print("=" * 65 + "\n")

    if args.dashboard:
        dash = DashboardGenerator(args.db)
        out_file = dash.generate_html("dashboard.html")
        print(f"\n[+] Interactive Dashboard updated: {out_file}")
        print(f"    Open in your browser: open {out_file}\n")

    if args.list:
        db = DataStore(args.db)
        sessions = db.get_all_sessions()
        print("\n" + "=" * 80)
        print(f"{'Session ID':<32} | {'Adversarial':<11} | {'Status':<14} | {'Goal':<20}")
        print("-" * 80)
        for s in sessions:
            adv_str = "YES (Malicious)" if s['is_adversarial'] else "NO (Benign)"
            goal = (s['primary_goal'][:22] + "...") if len(s['primary_goal']) > 22 else s['primary_goal']
            print(f"{s['session_id']:<32} | {adv_str:<11} | {s['status']:<14} | {goal:<20}")
        print("=" * 80 + "\n")

if __name__ == "__main__":
    main()
