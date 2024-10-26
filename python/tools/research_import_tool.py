import os
from agent import Agent
from python.helpers.tool import Tool, Response
from python.helpers import files
from python.tools import memory_tool

class ResearchImport(Tool):
    async def execute(self, **kwargs):
        try:
            work_dir = self.agent.get_data("work_dir") or os.getcwd()
            imported_count = 0
            
            # Get all .txt files in work_dir
            txt_files = [f for f in os.listdir(work_dir) if f.endswith('.txt')]
            
            for filename in txt_files:
                filepath = os.path.join(work_dir, filename)
                
                # Read the research file
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Generate a memory prompt using gpt-4o-mini
                prompt_system = "You are an AI assistant tasked with creating a concise prompt that describes the main topic and key points of a research document. This prompt will be used to store the research in a memory bank."
                prompt_request = f"Please create a brief prompt (2-3 sentences) that captures the essence of this research document:\n\n{content[:2000]}..."  # First 2000 chars for context
                
                memory_prompt = await self.agent.send_adhoc_message(
                    system=prompt_system,
                    msg=prompt_request,
                    output_label=f"Generating memory prompt for {filename}"
                )
                
                # Save to memory with the generated prompt
                await memory_tool.save(self.agent, f"Research Import - {memory_prompt}: {content}")
                imported_count += 1
            
            return Response(
                message=f"Successfully imported {imported_count} research files into memory.",
                break_loop=False
            )
            
        except Exception as e:
            return Response(
                message=f"Error importing research files: {str(e)}",
                break_loop=False
            )
