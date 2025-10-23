# Copyright (c) 2023 by Microsoft Corporation.
# Licensed under the MIT license.


import inspect
import collections
from typing import Optional, List

from parrot.sampling_config import SamplingConfig
from parrot.utils import get_logger, change_signature

from .semantic_variable import SemanticVariable
from .function import SemanticFunction, PyNativeFunction, ParamType, Parameter
from .transforms.prompt_formatter import standard_formatter, Sequential, FuncMutator


logger = get_logger("Interface")


# Annotations of arguments when defining a parrot function.


class Input:
    """Annotate the Input semantic variable in the Parrot function signature."""


class Output:
    """Annotate the Output semantic varialbe in the Parrot function signature."""

    def __init__(
        self,
        sampling_config: Optional[SamplingConfig] = None,
    ) -> None:
        # kw arguments for semantic functions
        self.sampling_config = sampling_config


def semantic_function(
    formatter: Optional[Sequential] = standard_formatter,
    conversation_template: Optional[FuncMutator] = None,
    try_register: bool = True,
    **semantic_func_metadata,
):
    """A decorator for users to define parrot functions."""

    def create_func(f):
        func_name = f.__name__
        doc_str = f.__doc__

        # print(doc_str)

        # Parse the function signature (parameters)
        func_sig = inspect.signature(f)
        return_annotations = func_sig.return_annotation
        func_params = []
        for param in func_sig.parameters.values():
            # assert param.annotation in (
            #     Input,
            #     Output,
            # ), "The arguments must be annotated by Input/Output"

            kwargs = {}

            if param.annotation == Input:
                param_typ = ParamType.INPUT_LOC
            elif param.annotation == Output:
                # Default output loc
                param_typ = ParamType.OUTPUT_LOC
                kwargs = {
                    "sampling_config": SamplingConfig(),
                }
            elif param.annotation.__class__ == Output:
                # Output loc with sampling config
                param_typ = ParamType.OUTPUT_LOC
                kwargs = {
                    "sampling_config": param.annotation.sampling_config,
                }
            else:
                param_typ = ParamType.INPUT_PYOBJ
            func_params.append(Parameter(name=param.name, typ=param_typ, **kwargs))

        if return_annotations != inspect.Signature.empty:
            raise ValueError("Semantic function can't not have return annotations.")

        semantic_func = SemanticFunction(
            name=func_name,
            params=func_params,
            func_body_str=doc_str,
            try_register=try_register,
            **semantic_func_metadata,
        )

        if formatter is not None:
            semantic_func = formatter.transform(semantic_func)
        if conversation_template is not None:
            logger.warning(
                f"Use a conversation template {conversation_template.__class__.__name__} to "
                "transform the function. This only works well for requests which are dispatched "
                "to engines with the corresponding models."
            )
            semantic_func = conversation_template.transform(semantic_func)

        return semantic_func

    return create_func


def native_function(
    **native_func_metadata,
):
    """A decorator for users to define parrot functions."""

    def create_func(f):
        func_name = f.__name__

        # Parse the function signature (parameters)
        func_sig = inspect.signature(f)
        return_annotations = func_sig.return_annotation
        func_params = []

        for param in func_sig.parameters.values():
            if param.annotation == Input:
                param_typ = ParamType.INPUT_LOC
            elif param.annotation == Output:
                # Default output loc
                param_typ = ParamType.OUTPUT_LOC
            elif param.annotation.__class__ == Output:
                # Output loc
                param_typ = ParamType.OUTPUT_LOC
            else:
                param_typ = ParamType.INPUT_PYOBJ
            func_params.append(Parameter(name=param.name, typ=param_typ))

        if return_annotations != inspect.Signature.empty:
            raise ValueError("Native function can't not have return annotations.")

        native_func = PyNativeFunction(
            name=func_name,
            pyfunc=f,
            params=func_params,
            # Func Metadata
            **native_func_metadata,
        )

        return native_func

    return create_func


def variable(
    name: Optional[str] = None, content: Optional[str] = None
) -> SemanticVariable:
    """Let user construct Semantic Variable explicitly."""

    return SemanticVariable(name, content)


# def shared_context(
#     engine_name: str,
#     parent_context: Optional[Context] = None,
# ) -> SharedContext:
#     """Interface to create a shared context."""

#     return SharedContext(engine_name, parent_context)
