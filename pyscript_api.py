import requests
import json
import logging

BASE_URL = "http://localhost:8765"
OPENPERPLEX_URL = "http://localhost:8082"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def process_response(response, endpoint_name):
    try:
        response.raise_for_status()
        result = response.json()
        return {"result": result.get('result', result)}
    except requests.RequestException as e:
        return {"error": f"Error in {endpoint_name}: {str(e)}"}
    except json.JSONDecodeError as e:
        return {"error": f"Error decoding JSON response in {endpoint_name}: {str(e)}"}
    except Exception as e:
        return {"error": f"Unexpected error in {endpoint_name}: {str(e)}"}

def run_agent(prompt, timeout=None):
    """Run an agent with a given prompt."""
    url = f"{BASE_URL}/run_agent"
    data = {
        "prompt": prompt,
        "timeout": timeout
    }

    response = requests.post(url, json=data, timeout=600)  # 10 minutes
    result = process_response(response, "run_agent")
    logger.info(result)
    return result

def run_agent_async(prompt, timeout=None):
    """Start an asynchronous agent task."""
    url = f"{BASE_URL}/run_agent_async"
    data = {
        "prompt": prompt,
        "timeout": timeout
    }

    response = requests.post(url, json=data, timeout=1200)  # 20 minutes
    result = process_response(response, "run_agent_async")
    logger.info(result)
    return result

def remember(text):
    """Save information to the agent's memory."""
    url = f"{BASE_URL}/remember"
    data = {"prompt": text}

    response = requests.post(url, json=data, timeout=180)
    result = process_response(response, "remember")
    logger.info(result)
    return result

def forget(prompt):
    """Remove information from the agent's memory."""
    url = f"{BASE_URL}/forget"
    data = {"prompt": prompt}

    response = requests.post(url, json=data, timeout=180)
    result = process_response(response, "forget")
    logger.info(result)
    return result

def recall(prompt, count=5, threshold=0.1):
    """Recall information from the agent's memory."""
    url = f"{BASE_URL}/recall"
    data = {
        "prompt": prompt,
        "count": count,
        "threshold": threshold
    }

    response = requests.post(url, json=data, timeout=180)
    result = process_response(response, "recall")
    logger.info(result)
    return result

def research(prompt=""):
    """Perform research on a given topic."""
    url = f"{BASE_URL}/research"
    data = {"prompt": prompt}

    response = requests.post(url, json=data, timeout=180)
    result = process_response(response, "research")
    logger.info(result)
    return result

def perplexity_search(prompt=""):
    """Perform a search using the Perplexity API."""
    url = f"{BASE_URL}/perplexity_search"
    data = {"prompt": prompt}

    response = requests.post(url, json=data, timeout=180)
    result = process_response(response, "perplexity_search")
    logger.info(result)
    return result

def openperplex_search(prompt="", date_context="", stored_location="", pro_mode=True):
    """Perform a search using the OpenPerplex API."""
    url = f"{OPENPERPLEX_URL}/search"
    params = {
        "query": prompt,
        "date_context": date_context,
        "stored_location": stored_location,
        "pro_mode": pro_mode
    }

    response = requests.get(url, params=params, timeout=300)
    result = process_response(response, "openperplex_search")
    logger.info(result)
    return result

def initialize():
    logger.info("API endpoints initialized")

if __name__ == "__main__":
    initialize()
