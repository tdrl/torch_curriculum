"""Just a demo that the App framework works."""

from torch_playground.util import App, BaseConfiguration
from dataclasses import dataclass, field, asdict
from structlog.dev import ConsoleRenderer

class DemoApp(App):
    """A simple demo application that extends the App framework."""

    @dataclass
    class DemoArguments(BaseConfiguration):
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
        super().__init__(self.DemoArguments(), description='Demo Application for Torch Playground',)
        self.logger.info('DemoApp initialized with arguments', **asdict(self.config))

    def run(self):
        """Run the demo application."""
        self.logger.info(f'Demo is running.')
        # Here you can add more functionality or logic for your demo app.
        self.logger.debug('This is a debug message.')
        self.logger.warning('This is a warning message.')
        self.logger.error('This is an error message.')
        self.logger.critical('This is a critical message.')
        self.logger.info('An info message with some structured data', key1='value1', key2=2)
        self.logger.info('Info message with dict structured data', **{'a': 1, 'b': 2})
        local_logger = self.logger.bind(local_key='local_value')
        local_logger.info('This is a message from the local logger.')
        try:
            x = 3
            y = 'Goodbye, cruel world!'
            raise RuntimeError('This is a simulated runtime error for demonstration purposes.')
        except RuntimeError as e:
            self.logger.exception('An exception occurred', exc_info=e)


if __name__ == '__main__':
    demo_app = DemoApp()
    demo_app.run()
