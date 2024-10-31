from langchain_community.utilities import DuckDuckGoSearchAPIWrapper

def search(query: str, results = 5, region = "wt-wt", time="y") -> list:
    # Create an instance with custom parameters
    api = DuckDuckGoSearchAPIWrapper(
        region=region,  # Set the region for search results
        safesearch="off",  # Set safesearch level (options: strict, moderate, off)
        time=time,  # Set time range (options: d, w, m, y)
    )
    # Perform a search
    result = api.results(query, max_results=results)
    return result

