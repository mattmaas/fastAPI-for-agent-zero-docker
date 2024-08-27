import requests
import json

BASE_URL = "http://localhost:8765"

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

@service(supports_response="only")
def run_agent(prompt, timeout=None):
    """Run an agent with a given prompt."""
    url = f"{BASE_URL}/run_agent"
    data = {
        "prompt": prompt,
        "timeout": timeout
    }
    
    response = task.executor(requests.post, url, json=data, timeout=180)
    result = process_response(response, "run_agent")
    log.info(result)
    return result

@service(supports_response="only")
def run_agent_async(prompt, timeout=None):
    """Start an asynchronous agent task."""
    url = f"{BASE_URL}/run_agent_async"
    data = {
        "prompt": prompt,
        "timeout": timeout
    }
    
    response = task.executor(requests.post, url, json=data, timeout=300)
    result = process_response(response, "run_agent_async")
    log.info(result)
    return result

@service(supports_response="only")
def remember(text):
    """Save information to the agent's memory."""
    url = f"{BASE_URL}/remember"
    data = {"prompt": text}
    
    response = task.executor(requests.post, url, json=data, timeout=180)
    result = process_response(response, "remember")
    log.info(result)
    return result

@service(supports_response="only")
def forget(prompt):
    """Remove information from the agent's memory."""
    url = f"{BASE_URL}/forget"
    data = {"prompt": prompt}
    
    response = task.executor(requests.post, url, json=data, timeout=180)
    result = process_response(response, "forget")
    log.info(result)
    return result

@service(supports_response="only")
def recall(prompt, count=5, threshold=0.1):
    """Recall information from the agent's memory."""
    url = f"{BASE_URL}/recall"
    data = {
        "prompt": prompt,
        "count": count,
        "threshold": threshold
    }
    
    response = task.executor(requests.post, url, json=data, timeout=180)
    result = process_response(response, "recall")
    log.info(result)
    return result

@service(supports_response="only")
def research(prompt=""):
    """Perform research on a given topic."""
    url = f"{BASE_URL}/research"
    data = {"prompt": prompt}
    
    response = task.executor(requests.post, url, json=data, timeout=180)
    result = process_response(response, "research")
    log.info(result)
    return result
    
@service(supports_response="only")
def perplexity_search(prompt=""):
    """Perform a search using the Perplexity API."""
    url = f"{BASE_URL}/perplexity_search"
    data = {"prompt": prompt}
    
    response = task.executor(requests.post, url, json=data, timeout=180)
    result = process_response(response, "perplexity_search")
    log.info(result)
    return result

@time_trigger("startup")
def initialize():
    log.info("API endpoints initialized")
