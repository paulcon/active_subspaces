{#- One page per public module; the member lists come from the module's __all__
    (autosummary_ignore_module_all = False), so the page and __all__ cannot drift. -#}
{%- set callables = functions + classes + exceptions -%}
{%- set data = members | reject("in", callables) | reject("in", modules | default([])) | list -%}
{{ fullname | escape | underline }}

{% if fullname.startswith("active_subspaces.experimental") %}
.. warning::

   Experimental. Import it explicitly with ``import active_subspaces.experimental``; the import
   issues an ``ExperimentalWarning`` (code ``experimental.module``) once per process. Its API
   and results may change in any release, without a deprecation period.

{% endif %}
.. automodule:: {{ fullname }}
   :no-members:

.. currentmodule:: {{ fullname }}

{% if functions %}
.. rubric:: Functions

.. autosummary::
   :toctree:
   :nosignatures:
{% for item in functions %}
   {{ item }}
{%- endfor %}
{% endif %}

{% if classes %}
.. rubric:: Classes

.. autosummary::
   :toctree:
   :nosignatures:
{% for item in classes %}
   {{ item }}
{%- endfor %}
{% endif %}

{% if exceptions %}
.. rubric:: Exceptions and warnings

.. autosummary::
   :toctree:
   :nosignatures:
{% for item in exceptions %}
   {{ item }}
{%- endfor %}
{% endif %}

{% if data %}
.. rubric:: Data

.. autosummary::
   :toctree:
{% for item in data %}
   {{ item }}
{%- endfor %}
{% endif %}
