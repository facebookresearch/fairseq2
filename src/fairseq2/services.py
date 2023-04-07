# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, Optional, Type, TypeVar, cast

_services: Dict[type, Any] = {}

T = TypeVar("T")


def get(kls: Type[T]) -> T:
    """Return the service registered for ``kls``."""
    try:
        return cast(T, _services[kls])
    except KeyError:
        raise RuntimeError(f"There is no service registered for type {kls}.")


def set(service: Any, kls: Optional[Type[T]] = None) -> None:
    """Register the specified service.

    :param service:
        The service to register.
    :param kls:
        If not ``None``, the service will be registered for ``kls`` instead of
        its own type.
    """
    if not kls:
        kls = type(service)
    else:
        if not isinstance(service, kls):
            raise ValueError(
                f"`service` must be of type {kls}, but is of type {type(service)} instead."
            )

    _services[kls] = service


def init() -> None:
    """Initialize all default services."""
    from fairseq2 import assets

    assets.init_services()


def clear() -> None:
    """Remove all registered services."""
    _services.clear()
