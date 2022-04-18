from inspect import issubclass

for attribute_name in dir(module):
        attribute = getattr(module, attribute_name)
        if isclass(attribute) and issubclass(attribute, PluginBase):
            globals()[attribute_name] = attribute
