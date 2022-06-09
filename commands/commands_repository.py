from commands.browser_commands import BrowserCommands
from commands.os_commands import OsCommands
from commands.base_commands import BaseCommands


class CommandsRepository:
    def __init__(self):
        self.browser_commands = BrowserCommands()
        self.os_commands = OsCommands()

    def execute(self, command: str):
        commands = self._find_command_place(command)
        if commands:
            commands.execute_command(command)

    def _find_command_place(self, command: str) -> BaseCommands:
        if self.browser_commands.in_command_dict(command):
            return self.browser_commands
        elif self.os_commands.in_command_dict(command):
            return self.os_commands
