# mqdtfit: A collection of Python function for empirical multichannel quantum defect calculations

The original code of mqdtfit, with theory and examples, has been published in Computer Physics Communications (CPC) in 2024: R M Potvliege, mqdtfit: A collection of Python functions for empirical multichannel quantum defect calculations, Comput. Phys. Comm. **300** (2024) 109172, https://doi.org/10.1016/j.cpc.2024.109172  [open access].

The files published with this article can be found in this repository, in the folder mqdtfit_cpc. They are also available from the CPC Library at https://doi.org/10.17632/nzgygzd96c.1 . 

These codes have been used, in particular, in new MQDT models of a number of series of strontium - see C L Vaillant, M P A Jones and R M Potvliege, Addendum: Multichannel quantum defect theory of strontium bound Rydberg states (2014 J. Phys. B: At. Mol. Opt. Phys. **47** 155001), J. Phys. B: At. Mol. Opt. Phys. **57** (2024) 199401, https://doi.org/10.1088/1361-6455/ad76f0 [open access]. The drivers used to generate these results can also be found in this repository, in the strontium folder.

## Contents

This repository currently contains the following items:


### The files published with the Computer Physics Communications article (the mqdt_cpc folder)

- **mqdtfit_cpc/mqdtfit.py**: The python source code of the module.
- **mqdtfit_cpc/UserManual.pdf**: Detailed description of the user-facing functions contained in this module. The user is referred to the article published in Computer Physics Communications for general information about these programs.
- **mqdtfit_cpc/strontium.py**: An example of program using the mqdtfit module.
- **mqdtfit_cpc/strontium_output.txt**: A copy of the text output generated when running strontium.py.
- **mqdtfit_cpc/channelfractions.pdf**: The graph produced by strontium.py.
- **mqdtfit_cpc/ytterbium.py**: Another example of program using the mqdtfit module.
- **mqdtfit_cpc/ytterbiumD2data.dat**: The data file read by ytterbium.py.
- **mqdtfit_cpc/ytterbium_output.txt**: A copy of the text output generated when running ytterbium.py.
- **mqdtfit_cpc/LuFanoplot.pdf**: The graph produced by ytterbium.py.

### The driver programs used to generate the strontium results published in the 2024 J. Phys. B article mentioned above (the strontium folder) 

These programs are meant to be used in conjunction with the mqdtfit.py module contained in the mqdt_cpc folder. 
