from dataclasses import dataclass
import os, json, contextlib, subprocess, ast, shlex, asyncio
from io import StringIO
import time
from typing import Literal
from python.helpers import files, messages
from agent import Agent
from python.helpers.tool import Tool, Response
from python.helpers import files
from python.helpers.print_style import PrintStyle
from python.helpers.shell_local import LocalInteractiveSession

@dataclass
class State:
    shell: LocalInteractiveSession

class CodeExecution(Tool):

    async def execute(self, **kwargs):
        if await self.agent.handle_intervention(): return Response(message="", break_loop=False)  # wait for intervention and handle it, if paused
        
        self.prepare_state()
        
        runtime = self.args["runtime"].lower().strip()
        if runtime == "python":
            response = await self.execute_python_code(self.args["code"])
        elif runtime == "nodejs":
            response = await self.execute_nodejs_code(self.args["code"])
        elif runtime == "terminal":
            response = self.execute_terminal_command(self.args["code"])
        elif runtime == "output":
            response = self.get_terminal_output()
        else:
            response = files.read_file("./prompts/fw.code_runtime_wrong.md", runtime=runtime)

        if not response: response = files.read_file("./prompts/fw.code_no_output.md")
        return Response(message=response, break_loop=False)

    async def after_execution(self, response, **kwargs):
        msg_response = files.read_file("./prompts/fw.tool_response.md", tool_name=self.name, tool_response=response.message)
        self.agent.append_message(msg_response, human=True)

    def prepare_state(self):
        self.state = self.agent.get_data("cot_state")
        if not self.state:
            shell = LocalInteractiveSession()
            self.state = State(shell=shell)
            shell.connect()
        self.agent.set_data("cot_state", self.state)
    
    async def execute_python_code(self, code):
        escaped_code = shlex.quote(code)
        command = f'python3 -c {escaped_code}'
        return await self.terminal_session(command)

    def execute_nodejs_code(self, code):
        escaped_code = shlex.quote(code)
        command = f'node -e {escaped_code}'
        return self.terminal_session(command)

    def execute_terminal_command(self, command):
        return self.terminal_session(command)

    async def terminal_session(self, command):
        if await self.agent.handle_intervention(): return ""  # wait for intervention and handle it, if paused
       
        self.state.shell.send_command(command)

        PrintStyle(background_color="white",font_color="#1B4F72",bold=True).print(f"{self.agent.agent_name} code execution output:")
        return await self.get_terminal_output()

    async def get_terminal_output(self):
        idle=0
        while True:       
            await asyncio.sleep(0.1)  # Wait for some output to be generated
            full_output, partial_output = self.state.shell.read_output()

            if await self.agent.handle_intervention(): return full_output  # wait for intervention and handle it, if paused
        
            if partial_output:
                PrintStyle(font_color="#85C1E9").stream(partial_output)
                idle=0    
            else:
                idle+=1
                if ( full_output and idle > 30 ) or ( not full_output and idle > 100 ): return full_output
                           
