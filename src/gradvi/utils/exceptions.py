
class NormalMeansInversionError(Exception):
    """
    Exceptions raised for errors during inverting
    the posterior mean of Normal Means model
    """
    
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)
