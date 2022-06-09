from webbrowser import Error

from speech_recognition.speech_recognition_flow import SpeechRecognitionFlow
from commands.commands_repository import CommandsRepository


class MainActivity:
    def __init__(self):
        self.flow = SpeechRecognitionFlow()
        self.command_repository = CommandsRepository()

    def run(self):
        try:
            self.flow.start()
            while True:
                command = self.flow.get_speech()
                self.command_repository.execute(command)
        except KeyboardInterrupt:
            pass
        except Error as e:
            print(e)
        finally:
            self.flow.stop()
            exit()

