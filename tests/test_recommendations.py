import sys, os
sys.path.append(os.getcwd())

from src.recommendations.service import recommend_products


def test_recommend_products_returns_results():
    results = recommend_products("travel rewards")
    assert isinstance(results, list)
    assert len(results) > 0
    first = results[0]
    assert "title" in first and "description" in first
