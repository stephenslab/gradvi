
import logging
import sys

from . import project

# Resources:
# https://stackoverflow.com/questions/39492471/how-to-extend-the-logging-logger-class
# https://www.toptal.com/python/in-depth-python-logging
# https://zetcode.com/python/logging/
# https://coralogix.com/blog/python-logging-best-practices-tips/


loggers = {}


def get_new_formatter(fmt = project.logging_format()):
    return logging.Formatter(fmt)


def get_new_handler(formatter, logfile = None):
    if logfile is None: # log to stdout
        handler = logging.StreamHandler(sys.stdout)
    else: # log to file
        handler = logging.FileHandler(logfile)
    handler.setFormatter(formatter)
    return handler


class CustomLogger(logging.getLoggerClass()):

    global loggers

    def __init__(self, name, fmt = None, level = None, logfile = None, is_handler = False, is_debug = False):
        # Make sure the project has a root logger
        if project.get_name() not in loggers.keys() and not name == project.get_name():
            self.create_default_logger()
        # Create the logger
        self.create(name, fmt = fmt, level = level, logfile = logfile, is_handler = is_handler, is_debug = is_debug)
        return

    def __repr__(self):
        if loggers.keys():
            m = max(map(len, list(loggers.keys()))) + 1
            return '\n'.join([k.rjust(m) + ':' + repr(v)
                              for k, v in loggers.items()])
        else:
            return self.__class__.__name__ + "()"


    def __dir__(self):
        return list(loggers.keys())


    def create_default_logger(self):
        self.create(project.get_name(), level = project.logging_level(),
            fmt = project.logging_format(), logfile = project.logging_file(),
            is_handler = True)
        return


    def override_global_default_loglevel(self, level):
        base_logger = loggers[project.get_name()]
        if level is None: level = base_logger.getEffectiveLevel()
        if not base_logger.getEffectiveLevel() == level:
            base_logger.setLevel(level)
        return


    def set_loglevel(self, level):
        if level is None: level = self.logger.parent.getEffectiveLevel()
        if not self.logger.getEffectiveLevel() == level:
            self.logger.setLevel(level)
        return


    def create(self, name, fmt = None, level = None, logfile = None, is_handler = False, is_debug = False):
        # Register new logger if not already present.
        # A logger is unique by name, meaning that if a logger with the name `foo` 
        # has been created, the consequent calls of `logging.getLogger("foo")` 
        # will return the same object.
        self.register_logger(name)

        # The log level can be altered during runtime.
        # Inherit parent logging level if level = None
        level = level if not is_debug else logging.DEBUG
        self.set_loglevel(level)

        # Python loggers form a hierarchy. A logger named main is a parent of main.new.
        # Child loggers propagate messages up to the handlers associated with their 
        # ancestor loggers. Because of this, it is unnecessary to define and configure 
        # handlers for all the loggers in the application. It is sufficient to 
        # configure handlers for a top-level logger and create child loggers as needed.
        # 
        # If a child logger gets a new handler, then the same information will be 
        # processed by the handlers of the child logger and the parent logger.
        if is_handler:
            formatter = get_new_formatter(fmt)
            handler   = get_new_handler(formatter, logfile)
            self.logger.addHandler(handler)
        return


    def info(self, msg, extra=None):
        self.logger.info(msg, extra=extra)


    def error(self, msg, extra=None):
        self.logger.error(msg, extra=extra)


    def debug(self, msg, extra=None):
        self.logger.debug(msg, extra=extra)


    def warn(self, msg, extra=None):
        self.logger.warning(msg, extra=extra)


    def register_logger(self, name):
        self.logger = loggers.get(name)
        if not self.logger:
            self.logger = logging.getLogger(name)
            loggers[name] = self.logger
            self.logger.debug(f"Created {self.logger}, Parent: {self.logger.parent}")
        #else:
        #    self.logger.debug(f"Using old {self.logger}")
        return
