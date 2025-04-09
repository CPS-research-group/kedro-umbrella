# Pipe deformation use case

To run the pipeline, use the Makefile accordingly. Different rules are provided.

This example focuses on *structural health monitoring*. The goal is to predict the plastic strain of a certain structure given an observed deformation.

The pipeline proceeds as follows:

1. Load finite element (FE) data simulate different impact strengths and obtain deformation `displ` and plastic strain `eps` values as output.
2. Reduce the deformations and plastic strains using principal component analysis (PCA): one PCA per each with Coder Box.
3. Encode the data in PCA reduced dimensions.
4. Train a regression model to predict the reduced plastic strain from the reduced deformation.
