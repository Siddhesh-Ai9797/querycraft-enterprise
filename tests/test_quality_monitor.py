import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.quality_monitor import QualityMonitor

def test_high_quality():
    monitor = QualityMonitor()
    result  = monitor.score(
        {"success": True, "confidence": 1.0, "sql": "SELECT COUNT(*) FROM artist"},
        {"row_count": 1, "success": True}
    )
    assert result["quality_score"] >= 0.9
    assert result["flagged"] is False
    print(f"✅ High quality score: {result['quality_score']}")

def test_failed_pipeline():
    monitor = QualityMonitor()
    result  = monitor.score({"success": False}, {"row_count": 0, "success": False})
    assert result["quality_score"] == 0.0
    assert result["flagged"] is True
    print(f"✅ Failed pipeline flagged correctly")

def test_complexity_detection():
    monitor = QualityMonitor()
    simple  = monitor._score_complexity("SELECT COUNT(*) FROM artist")
    medium  = monitor._score_complexity("SELECT * FROM artist a JOIN album b ON a.artist_id = b.artist_id")
    assert simple[0]  == "simple"
    assert medium[0]  == "medium"
    print(f"✅ Complexity: simple={simple[0]}, medium={medium[0]}")

if __name__ == "__main__":
    test_high_quality()
    test_failed_pipeline()
    test_complexity_detection()