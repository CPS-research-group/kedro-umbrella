# DesCartes Builder: Back-end implementation

## Overview

This package provides facilitates to specify ML pipeline as employed in Digital Twin design. This is a part of the DesCartes Builder, see [https://descartes.cnrsatcreate.cnrs.fr/wp9-augmented-engineering/]. Digital Twin (DT) often have (1) complex & diverse pipelines, but despite their variety, there are common building blocks. Besides, (2) DTs are often composed of several functions.

The Function+Data Flow (FDF) implemented here extends a traditional data-flow in two ways:

1. incorporating functions as first-class citizens: FDF allows know whether a function or data is generated, allowing to reuse/export functions explicitly
2. defining application-specific boxes representing different processing steps that are required for DT design.

This package is implemented leveraging Kedro for executing and describing the pipeline is a simple way. We extend the Kedro `Node` to implement the FDF boxes and we can use Kedro pipeline as is for execution. We also include a library of commonly used functions for each application-specific code.

## Installation & development

Here are instructions to install the library. Using a `conda` environment is highly recommended.

```bash
# Clone repo
git clone git@github.com:CPS-research-group/kedro-umbrella.git
# Create development env
conda create -n builder_dev python=3.10.8
conda activate builder_dev
# Install library
make install
```

Tests for key features are in `./tests`: execute with `make test`.

## Examples

Some examples and case studies using FDF are in the `./examples` folder. Each example is documented in its own folder in a README.md file. Different variants of the case study pipeline can be executed with Makefile rules (briefly described in the Makefile itself):

- Run all examples: `cd examples && make`.
- Run a single example: `cd examples/<example_dir> && make <example_rule>`
