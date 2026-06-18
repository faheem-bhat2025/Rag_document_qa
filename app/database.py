import sqlite3

class Database:
    def __init__(self, DB_PATH):
        self.DB_PATH = DB_PATH
        """Create the logs table if it doesn't exist. Safe to call repeatedly."""
        conn = sqlite3.connect(self.DB_PATH)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS query_logs (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                query       TEXT    NOT NULL,
                answer      TEXT,
                no_answer   INTEGER NOT NULL DEFAULT 0,
                top_score   REAL,
                latency_ms  REAL    NOT NULL,
                created_at  TEXT    NOT NULL DEFAULT (datetime('now'))
            )
        """)
        conn.commit()
        conn.close()


    def log_query(self,query, answer, no_answer, top_score, latency_ms):
        """Insert one row recording an /ask interaction."""
        conn = sqlite3.connect(self.DB_PATH)
        conn.execute(
            "INSERT INTO query_logs (query, answer, no_answer, top_score, latency_ms) "
            "VALUES (?, ?, ?, ?, ?)",
            (query, answer, int(no_answer), float(top_score), float(latency_ms)),
        )
        conn.commit()
        conn.close()


    def get_analytics(self):
        """Run the three required analytics queries and return them as a dict."""
        conn = sqlite3.connect(self.DB_PATH)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()

        cur.execute("""
            SELECT query, COUNT(*) AS times_asked
            FROM query_logs
            GROUP BY query
            ORDER BY times_asked DESC
            LIMIT 5
        """)
        most_frequent = [dict(r) for r in cur.fetchall()]

        cur.execute("""
            SELECT query, created_at
            FROM query_logs
            WHERE no_answer = 1
            ORDER BY created_at DESC
        """)
        no_answer_queries = [dict(r) for r in cur.fetchall()]

        cur.execute("SELECT AVG(latency_ms) AS avg_latency_ms FROM query_logs")
        avg_latency = cur.fetchone()["avg_latency_ms"] or 0.0

        cur.execute("SELECT COUNT(*) AS total FROM query_logs")
        total = cur.fetchone()["total"]

        conn.close()
        return {
            "total_queries": total,
            "most_frequent_questions": most_frequent,
            "no_answer_queries": no_answer_queries,
            "average_latency_ms": round(avg_latency, 2),
        }