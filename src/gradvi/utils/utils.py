
def get_optional_arg(key, default, **kwargs):
    x = kwargs[key] if key in kwargs.keys() else default
    return x
