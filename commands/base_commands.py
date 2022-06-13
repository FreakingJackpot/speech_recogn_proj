from abc import ABCMeta, abstractmethod


class BaseCommands(metaclass=ABCMeta):

    @abstractmethod
    def __init__(self):
        self._command_dict = {}

    def in_command_dict(self, command_str: str) -> bool:
        return command_str in self._command_dict

    def execute_command(self, command_str: str) -> None:
        command = self._command_dict.get(command_str)
        command()
