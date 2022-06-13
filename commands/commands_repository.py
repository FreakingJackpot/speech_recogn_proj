from typing import Tuple

from commands.browser_commands import BrowserCommands
from commands.os_commands import OsCommands
from commands.base_commands import BaseCommands


class CommandsRepository:
    def __init__(self):
        self.commands_storages: Tuple[BaseCommands] = (BrowserCommands(), OsCommands())

    def execute(self, command: str):
        commands = self._find_command_place(command)
        if commands:
            commands.execute_command(command)

    def _find_command_place(self, command: str) -> BaseCommands:
        for storage in self.commands_storages:
            if storage.in_command_dict(command):
                return storage
