from agent import Agent
from python.helpers import perplexity_search
from python.helpers.tool import Tool, Response

class OnlineKnowledge(Tool):
    def execute(self,**kwargs):
        return Response(
            message=process_prompt(self.args["prompt"]),
            break_loop=False,
        )

def process_prompt(prompt):
    return str(perplexity_search.perplexity_search(prompt))
