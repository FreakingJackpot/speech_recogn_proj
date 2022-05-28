from webbrowser import Error

from speech_recognition.speech_recognition_flow import SpeechRecognitionFlow
from commands.command_repository import CommandRepository


def main():
    flow = SpeechRecognitionFlow()
    command_repository = CommandRepository()
    try:
        flow.start()
        while True:
            command = flow.get_speech()
            print(command)
            command_repository.execute(command)
    except KeyboardInterrupt:
        pass
    except Error as e:
        print(e)
    finally:
        flow.stop()
        exit()


if __name__ == '__main__':
    main()
