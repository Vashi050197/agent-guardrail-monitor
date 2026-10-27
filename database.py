"""
Database persistence layer for the Weak-to-Strong Agent Monitor & Guardrail.
Stores agent trajectories, reasoning traces (CoT), tool executions, and multi-tier monitor audits.
"""

import sqlite3
import json
from datetime import datetime
from typing import Dict, List, Optional, Any

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    agent_model TEXT NOT NULL,
    monitor_model TEXT NOT NULL,
    primary_goal TEXT NOT NULL,
    is_adversarial INTEGER DEFAULT 0,
    evasion_strategy TEXT DEFAULT 'none',
    status TEXT DEFAULT 'RUNNING',  -- 'RUNNING', 'COMPLETED', 'CIRCUIT_BROKEN'
    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    ended_at TIMESTAMP
);

CREATE TABLE IF NOT EXISTS steps (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    step_num INTEGER NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    cot_reasoning TEXT NOT NULL,
    tool_name TEXT NOT NULL,
    tool_args TEXT NOT NULL,
    tool_output TEXT NOT NULL,
    FOREIGN KEY(session_id) REFERENCES sessions(session_id)
);

CREATE TABLE IF NOT EXISTS audit_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    step_num INTEGER NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    hierarchical_score REAL NOT NULL,
    sequential_score REAL NOT NULL,
    hybrid_score REAL NOT NULL,
    cot_faithfulness_score REAL NOT NULL,
    guardrail_decision TEXT NOT NULL, -- 'ALLOW', 'FLAG_WARNING', 'CIRCUIT_BREAKER'
    monitor_reasoning TEXT NOT NULL,
    rolling_summary TEXT,
    FOREIGN KEY(session_id) REFERENCES sessions(session_id)
);

CREATE TABLE IF NOT EXISTS time_series_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    total_steps INTEGER NOT NULL,
    avg_suspicion_last_10 REAL NOT NULL,
    avg_faithfulness_last_10 REAL NOT NULL,
    anomaly_rate REAL NOT NULL,
    circuit_breaker_trips INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_steps_session ON steps(session_id, step_num);
CREATE INDEX IF NOT EXISTS idx_audit_session ON audit_logs(session_id, step_num);
"""

class DataStore:
    def __init__(self, db_path: str = "agent_monitor.db"):
        self.db_path = db_path
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._get_conn() as conn:
            conn.executescript(SCHEMA_SQL)

    def create_session(self, session_id: str, agent_model: str, monitor_model: str,
                       primary_goal: str, is_adversarial: bool = False,
                       evasion_strategy: str = "none"):
        with self._get_conn() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO sessions 
                   (session_id, agent_model, monitor_model, primary_goal, is_adversarial, evasion_strategy)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (session_id, agent_model, monitor_model, primary_goal, 1 if is_adversarial else 0, evasion_strategy)
            )

    def log_step(self, session_id: str, step_num: int, cot_reasoning: str,
                 tool_name: str, tool_args: Dict[str, Any], tool_output: str) -> int:
        with self._get_conn() as conn:
            cur = conn.execute(
                """INSERT INTO steps (session_id, step_num, cot_reasoning, tool_name, tool_args, tool_output)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (session_id, step_num, cot_reasoning, tool_name, json.dumps(tool_args), tool_output)
            )
            return cur.lastrowid

    def log_audit(self, session_id: str, step_num: int, hier_score: float,
                  seq_score: float, hybrid_score: float, faithfulness_score: float,
                  decision: str, reasoning: str, rolling_summary: str):
        with self._get_conn() as conn:
            conn.execute(
                """INSERT INTO audit_logs 
                   (session_id, step_num, hierarchical_score, sequential_score, 
                    hybrid_score, cot_faithfulness_score, guardrail_decision, monitor_reasoning, rolling_summary)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (session_id, step_num, hier_score, seq_score, hybrid_score,
                 faithfulness_score, decision, reasoning, rolling_summary)
            )
            # If circuit breaker, mark session
            if decision == "CIRCUIT_BREAKER":
                conn.execute("UPDATE sessions SET status = 'CIRCUIT_BROKEN', ended_at = CURRENT_TIMESTAMP WHERE session_id = ?",
                             (session_id,))

    def complete_session(self, session_id: str, status: str = "COMPLETED"):
        with self._get_conn() as conn:
            conn.execute("UPDATE sessions SET status = ?, ended_at = CURRENT_TIMESTAMP WHERE session_id = ?",
                         (status, session_id))

    def record_snapshot(self):
        with self._get_conn() as conn:
            # Calculate rolling window metrics
            cur = conn.execute("""
                SELECT 
                    AVG(hybrid_score) as avg_suspicion,
                    AVG(cot_faithfulness_score) as avg_faith,
                    SUM(CASE WHEN guardrail_decision != 'ALLOW' THEN 1 ELSE 0 END) * 1.0 / COUNT(*) as anomaly_rate,
                    SUM(CASE WHEN guardrail_decision = 'CIRCUIT_BREAKER' THEN 1 ELSE 0 END) as trips,
                    COUNT(*) as total_steps
                FROM (SELECT * FROM audit_logs ORDER BY id DESC LIMIT 10)
            """)
            row = cur.fetchone()
            if row and row['total_steps'] and row['total_steps'] > 0:
                conn.execute(
                    """INSERT INTO time_series_snapshots 
                       (total_steps, avg_suspicion_last_10, avg_faithfulness_last_10, anomaly_rate, circuit_breaker_trips)
                       VALUES (?, ?, ?, ?, ?)""",
                    (row['total_steps'], row['avg_suspicion'] or 1.0, row['avg_faith'] or 1.0,
                     row['anomaly_rate'] or 0.0, row['trips'] or 0)
                )

    def get_session_history(self, session_id: str) -> List[Dict]:
        with self._get_conn() as conn:
            cur = conn.execute("""
                SELECT s.step_num, s.timestamp, s.cot_reasoning, s.tool_name, s.tool_args, s.tool_output,
                       a.hierarchical_score, a.sequential_score, a.hybrid_score, 
                       a.cot_faithfulness_score, a.guardrail_decision, a.monitor_reasoning
                FROM steps s
                LEFT JOIN audit_logs a ON s.session_id = a.session_id AND s.step_num = a.step_num
                WHERE s.session_id = ?
                ORDER BY s.step_num ASC
            """, (session_id,))
            return [dict(r) for r in cur.fetchall()]

    def get_all_sessions(self) -> List[Dict]:
        with self._get_conn() as conn:
            cur = conn.execute("SELECT * FROM sessions ORDER BY started_at DESC")
            return [dict(r) for r in cur.fetchall()]
