.. _HowToUse:
==================
How to use eTraGo?
==================

After you installed eTraGo you would typically start optimization runs by
executing the ‘appl.py’ which is situated in 
``./eTrago/etrago/`` (e.g by ``python3 appl.py``).

eTraGo doesn't have a graphical user interface, 
the ‘appl.py’ is used as a simple user interface which can be edited with 
the preferred python-editor.
Here parameters, calculation methods and scenario settings are set in a python
dictionary called 'args'. 
To run the desired calculation, it is crucial to understand these parameters. 
In addition, some of them contradict the usage of others. 
You find the documentation of all defined parameters from the 'args' here:
:func:`etrago.appl.etrago`.

Alternatively, the 'args' dictionary can be edited in a json-file.
Then the path to the json-file has to be defined in the function
:meth:`etrago.tools.utilities.get_args_setting`. Once a path is given
and the get_args_setting() within the `'appl.py' <https://github.com/openego/eTraGo/blob/37a91c92fd9eafc31bd0679334c906ac571a2b18/etrago/appl.py#L144>`_
is executed the 'args' dictionary within the 'appl.py' is ignored
and replaced by the 'args' of the json-file.

The appl.py contains the :func:`etrago.appl.etrago` function which uses the
defined 'args' dictionary to start the desired calculation.

To improve the performance of the optimization of the selected solver, 
you might want to use solver options (part of 'args'). For gurobi
the most used ones are described 
`here <https://github.com/openego/eTraGo/issues/213>`_.

Moreover, if you want to change parameters apart from the options which
are provided by the 'args' you can change the default values of 
the arguments used in the functions which are executed by the 
:meth:`etrago.appl.etrago` function.
Lastly, for more specific or extensive changes you are highly invited
to write code and add new functionalities.

Once the calculation has finished a PyPSA network will contain all results. 
You can use several plotting functions from the :meth:`etrago.tools.plot` in order
to visualize the results. For example 
the :meth:`etrago.tools.plot.plot_line_loading` plots
the relative line loading in % of all AC lines and DC links of the network.

To save the results you can use an interface to the oedb or write them
simply to csv files. These functionalites can be specified 
also in :meth:`etrago.appl.etrago`.


.. _Examples:
Examples and tutorial notebooks
===============================



.. toctree::
   :maxdepth: 7

   OpenMod  <https://github.com/openego/eGo/blob/master/ego/examples/tutorials/etrago_OpenMod_Zuerich18.ipynb>
