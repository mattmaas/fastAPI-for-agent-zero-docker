import requests
import json

BASE_URL = "http://localhost:8766"

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
def run_agent(prompt):
    """Run an agent thread with a given prompt."""
    url = f"{BASE_URL}/run_agent"
    data = {
        "prompt": prompt
    }

    response = task.executor(requests.post, url, json=data)
    result = process_response(response, "run_agent")
    log.info(result)
    return result

@service(supports_response="only")
def run_agent_async(prompt):
    """Start an asynchronous agent task thread."""
    url = f"{BASE_URL}/run_agent_async"
    data = {
        "prompt": prompt
    }

    response = task.executor(requests.post, url, json=data)
    result = process_response(response, "run_agent_async")
    log.info(result)
    return result

@service(supports_response="only")
def remember(prompt):
    """Save information to the your memory."""
    url = f"{BASE_URL}/remember"
    data = {"prompt": prompt}

    response = task.executor(requests.post, url, json=data)
    result = process_response(response, "remember")
    log.info(result)
    return result

@service(supports_response="only")
def forget(prompt):
    """Remove information from your memory. Forget memories."""
    url = f"{BASE_URL}/forget"
    data = {"prompt": prompt}

    response = task.executor(requests.post, url, json=data)
    result = process_response(response, "forget")
    log.info(result)
    return result

@service(supports_response="only")
def recall(prompt, count=5, threshold=0.1):
    """Recall information from the your memory on a given topic."""
    url = f"{BASE_URL}/recall"
    data = {
        "prompt": prompt,
        "count": count,
        "threshold": threshold
    }

    response = task.executor(requests.post, url, json=data)
    result = process_response(response, "recall")
    log.info(result)
    return result

@service(supports_response="only")
def research(prompt):
    """Recall memories & Perform research on a given topic. (Updates memory)"""
    url = f"{BASE_URL}/research"
    data = {"prompt": prompt}

    response = task.executor(requests.post, url, json=data)
    result = process_response(response, "research")
    log.info(result)
    return result

@service(supports_response="only")
def research_lite(prompt):
    """Recall memories & Perform research with less sources and tokens used on a given topic.  (Updates memory)"""
    url = f"{BASE_URL}/research_lite"
    data = {"prompt": prompt}

    response = task.executor(requests.post, url, json=data)
    result = process_response(response, "research_lite")
    log.info(result)
    return result

@service(supports_response="only")
def perplexity_search(prompt):
    """Recall memories & Perform a search using the Perplexity API. (Updates memory)"""
    url = f"{BASE_URL}/perplexity_search"
    data = {"prompt": prompt}

    response = task.executor(requests.post, url, json=data)
    result = process_response(response, "perplexity_search")
    log.info(result)
    return result

@service(supports_response="only")
def wolfram_alpha_search(prompt):
    """Only call this function if requested by the user. Perform a Wolfram Alpha search on a given topic."""
    url = f"{BASE_URL}/wolfram_alpha_search"
    data = {"prompt": prompt}

    response = task.executor(requests.post, url, json=data)
    result = process_response(response, "wolfram_alpha_search")
    log.info(result)
    return result

@service(supports_response="only")
def youtube_search(prompt):
    """Only call this function if requested by the user. Unless dircted otherwise, return the exact URLs found for the YouTube videos at the top of your answer to the user."""
    url = f"{BASE_URL}/youtube_search"
    data = {"prompt": prompt}

    response = task.executor(requests.post, url, json=data)
    result = process_response(response, "youtube_search")
    log.info(result)
    return result

@service(supports_response="only")
def reddit_search(prompt):
    """Only call this function if requested by the user. Unless dircted otherwise, return the exact URLs found for the Reddit posts at the top of your answer to the user."""
    url = f"{BASE_URL}/reddit_search"
    data = {"prompt": prompt}

    response = task.executor(requests.post, url, json=data)
    result = process_response(response, "reddit_search")
    log.info(result)
    return result

@time_trigger("startup")
def initialize():
    log.info("API endpoints initialized")
