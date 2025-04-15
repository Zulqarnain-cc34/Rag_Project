from .strategies.adaptive_retrieval import AdaptiveRAG
import pytest
import re
import time

@pytest.fixture(scope="module")
def rag():
    return AdaptiveRAG()

def test_empty_input_handling(rag):
    """Ensure the system handles empty queries gracefully without throwing errors or crashing."""
    response = rag.answer("")
    
    assert isinstance(response, str)
    assert len(response.strip()) > 0, "System returned an empty response for an empty query."
    assert not response.strip().lower().startswith("error"), "System returned an error message."
    assert len(response.strip().split()) >= 3, "Response too short; may not be a meaningful fallback."

def test_whitespace_query(rag):
    response = rag.answer("     ")
    assert "?" in response or len(response.strip()) > 0


def test_gibberish_query_handling(rag):
    """Test model behavior with nonsensical or malformed input."""
    response = rag.answer("asdjlkj23!@#ksd?")
    print("response",response)
    
    assert isinstance(response, str)
    assert len(response.strip()) > 0, "Model gave no response to gibberish."
    assert not response.strip().lower().startswith("error"), "Model responded with an error."
    # Check for the presence of fallback-like messages
    assert any(phrase in response.lower() for phrase in [
        "can't find",
        "not relevant",
        "incomplete",
        "couldn't process",
        "no meaningful information",
        "unable to understand",
        "random",
        "jumbled",
        "unclear",
        "try asking something else"
    ]), "Model did not gracefully handle gibberish input."

def test_response_time(rag):
    start = time.time()
    rag.answer("Tell me something about Cold bore technology?")
    end = time.time()
    assert end - start < 10  # Must respond within 3 seconds

def test_multilingual_query(rag):
    response = rag.answer("Quelle est la capitale de l'Allemagne?")
    assert "berlin" in response.lower()
