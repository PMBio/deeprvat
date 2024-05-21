.. DeepRVAT documentation master file, created by
   sphinx-quickstart on Wed Nov 22 10:24:36 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to DeepRVAT's documentation!
======================================

Rare variant association testing using deep learning and data-driven burden scores.


How to use this documentation
===================================

A good place to start is in :doc:`Basic usage </quickstart>`, to install the package and make sure it runs correctly.

To run DeepRVAT on your data, first consult *Modes of usage* :doc:`here </practical>`, then proceed based on which mode is right for your use case.

For all modes, you'll want to consult *Input data: Common requirements for all pipelines* and *Configuration file: Common parameters* :doc:`here </deeprvat>`.

For all modes of usage other than association testing with precomputed burdens, you'll need to :doc:`preprocess </preprocessing>` your genotype data, followed by :doc:`annotating </annotations>` your variants.

To train custom DeepRVAT models, rather than using precomputed burdens or our provided pretrained models, you'll need to additionally run :doc:`seed gene discovery </seed_gene_discovery>`.

Finally, consult the relevant section for your use case :doc:`here </deeprvat>`.

If running DeepRVAT on a cluster (recommended), some helpful tips are :doc:`here </cluster>`.


Citation
====================================

If you use this package, please cite:

Clarke, Holtkamp et al., “Integration of Variant Annotations Using Deep Set Networks Boosts Rare Variant Association Genetics.” bioRxiv. https://dx.doi.org/10.1101/2023.07.12.548506


Contact
====================================

To report a bug or make a feature request, please create an `issue <https://github.com/PMBio/deeprvat/issues>`_ on GitHub.

| For general inquiries, please contact:
| brian.clarke@dkfz.de
| eva.holtkamp@cit.tum.de


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation.md
   quickstart.md
   preprocessing.md
   annotations.md
   seed_gene_discovery.md
   deeprvat.md
   cluster.md
   practical.md
   ukbiobank.md
   apidocs/index


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
