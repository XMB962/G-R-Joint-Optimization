

class Registry(object):
    """

    To create a registry (e.g. a backbone registry):

    .. code-block:: python

        BACKBONE_REGISTRY = Registry('BACKBONE')

    To register an object:

    .. code-block:: python

        @BACKBONE_REGISTRY.register()
        class MyBackbone():
            ...

    Or:

    .. code-block:: python

        BACKBONE_REGISTRY.register(MyBackbone)
    """

    def __init__(self, name: str) -> None:
        """
        Args:
            name (str): the name of this registry
        """
        self._name: str = name
        self._obj_map: Dict[str, object] = {}

    def _do_register(self, name: str, obj: object) -> None:
        assert (
            name not in self._obj_map
        ), "An object named '{}' was already registered in '{}' registry!".format(
            name, self._name
        )
        self._obj_map[name] = obj

    def register(self, obj: object = None):
        """
        Register the given object under the the name `obj.__name__`.
        Can be used as either a decorator or not. See docstring of this class for usage.
        """
        if obj is None:
            # used as a decorator
            def deco(func_or_class: object) -> object:
                name = func_or_class.__name__  # pyre-ignore
                self._do_register(name, func_or_class)
                return func_or_class

            return deco

        name = obj.__name__  # pyre-ignore
        print("name: ", name)
        self._do_register(name, obj)  # 添加字典

    def get(self, name: str) -> object:
        ret = self._obj_map.get(name)
        if ret is None:
            raise KeyError(
                "No object named '{}' found in '{}' registry!".format(
                    name, self._name)
            )
        return ret

    def __contains__(self, name: str) -> bool:
        return name in self._obj_map


