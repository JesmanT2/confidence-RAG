from app.pipeline import build_service


def test_gate_abstains_when_context_is_insufficient():
    service = build_service()
    result = service.answer_query("Does this policy include crypto theft reimbursement?", threshold=0.55)

    assert result["abstained"] is True
    assert "don't have enough information" in result["answer"].lower()
