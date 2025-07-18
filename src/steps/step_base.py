class Step:
    """
    Abstract base class for all pipeline steps.

    Each subclass must implement the `run()` method.
    Provides access to shared configuration and logger and exposes the class name via the `name` property.
    """

    def __init__(self, cfg, logger):
        self.cfg = cfg # configuration dictionary passed to the step
        self.logger = logger # logger instance

    @property
    def name(self): 
        """Returns the class name of the step (used as the step name)."""
        return self.__class__.__name__
    
    def run(self): 
        """Subclasses must override this method to perform their step logic."""
        raise NotImplementedError()