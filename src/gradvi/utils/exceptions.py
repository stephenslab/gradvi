
class NMInversionError(Exception):
    """
    Exceptions raised for errors during inverting
    the posterior mean of Normal Means model
    """
    
    def __init__(self, method, message, is_diverging = False):
        self.method = method
        self.message = message
        self.is_diverging = is_diverging
        super().__init__(self.message)

    def __str__(self):
        return f'Error in inverting the posterior mean of Normal Means Model\n{self.method} -- {self.message}'
