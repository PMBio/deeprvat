.. DeepRVAT documentation master file, created by
   sphinx-quickstart on Wed Nov 22 10:24:36 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to DeepRVAT's documentation!
======================================

Rare variant association testing using deep learning and data-driven gene impairment scores.

_Coming soon:_ Overview of the DeepRVAT methodaster


How to use this documentation
===================================

First, see :doc:`Installation </installation>`.

Visit :doc:`Quick start </quickstart>` if you want to skip the detailed documentation and dive right in, or if you want to quickly check that the package installed correctly.

To run DeepRVAT on your data, first consult :doc:`Modes of usage </modes_of_usage>` then visit the section relevant to your use case.

For all modes, you'll want to consult :doc:`Input data and configuration </input_data>`.

Note also that for all modes of usage other than association testing with precomputed gene impairment scores, you'll need to :doc:`preprocess </preprocessing>` your genotype data, followed by :doc:`annotating </annotations>` your variants.

To train custom DeepRVAT models, rather than using precomputed gene impairment scores or our provided pretrained models, you'll need to additionally run :doc:`seed gene discovery </seed_gene_discovery>`. See also the :doc:`Practical recommendations for training </practical>`.

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
   modes_of_usage.md
   preprocessing.md
   annotations.md
   seed_gene_discovery.md
   precomputed_burdens.md
   pretrained_models.md
   training_association.md
   input_data.md
   output_files.md
   cluster.md
   practical.md
   ukbiobank.md
   apidocs/index


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
