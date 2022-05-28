from commands.browser_commands import BrowserCommands
from commands.os_commands import OsCommands


class CommandRepository:
    def __init__(self):
        self.browser_commands = BrowserCommands()
        self.os_commands = OsCommands()

    def execute(self, command: str):
        match command:
            case "открыть браузер":
                self.browser_commands.open_browser()
            case "курс валют":
                self.browser_commands.open_currency_page()
            case "выключить компьютер":
                self.os_commands.shutdown()
            case "режим сна":
                self.os_commands.suspend()
            case "увеличить громкость":
                self.os_commands.volume_up()
            case "уменьшить громкость":
                self.os_commands.volume_down()
            case "выключить звук":
                self.os_commands.mute()
            case "включить звук":
                self.os_commands.unmute()
            case "открыть ножницы":
                self.os_commands.open_scissors()
            case "открыть календарь":
                self.os_commands.open_calendar()
            case "открыть калькулятор":
                self.os_commands.open_calculator()
            case _:
                pass
