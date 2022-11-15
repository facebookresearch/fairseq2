.. currentmodule:: {{ module }}

{{ name | escape | underline }}

{% if name != "IList" and name != "IDict" %}
.. autoclass:: {{ name }}
    :members:
    :member-order: groupwise
    :class-doc-from: both
    :special-members: __call__, __iter__
    :inherited-members: Module
    :show-inheritance:
{% else %}
.. autoclass:: {{ name }}
    :no-members:
    :class-doc-from: class
    :show-inheritance:
{% endif %}
