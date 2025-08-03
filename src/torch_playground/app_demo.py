"""Just a demo that the App framework works."""

from torch_playground.util import App, BaseArguments
from dataclasses import dataclass, field, asdict
from structlog.dev import ConsoleRenderer

class DemoApp(App):
    """A simple demo application that extends the App framework."""

    @dataclass
    class DemoArguments(BaseArguments):
        """Command line arguments for the demo application."""
        fnord: str = field(default='Now you see me',
                           metadata={
                               'help': 'If you know, you know. (default: %(default)s)'
                           })
        magic: int = field(default=42,
                           metadata={
                               'help': 'The answer to life, the universe, and everything. (default: %(default)s)'
                           })

    def __init__(self):
        """Initialize the demo application with command line arguments."""
        super().__init__(self.DemoArguments())
        self.logger.info('DemoApp initialized with arguments', **asdict(self.args))

    def run(self):
        """Run the demo application."""
        self.logger.info('DemoApp is running.')
        # Here you can add more functionality or logic for your demo app.
        self.logger.debug('This is a debug message.')
        self.logger.warning('This is a warning message.')
        self.logger.error('This is an error message.')
        self.logger.critical('This is a critical message.')


if __name__ == '__main__':
    demo_app = DemoApp()
    demo_app.run()
