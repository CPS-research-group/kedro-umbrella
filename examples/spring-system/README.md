# Spring system use case 

To run the pipeline, use the Makefile accordingly. Different rules are provided. 

This example focuses on a simple spring system. The goal is to predict the spring stiffness given an observed displacement.

The pipeline proceeds as follows:
1. Simulate data from the spring system with different spring stiffness and obtain the corresponding displacement. Simulation is therefore stiffness -> displacement.  For ML, we want to perform the inverse problem displacement -> stiffness. In `get_data` we call `X` displacement and `Y` stiffness.
2. Encode both `X` and `Y` with Code blocks and reuse this function in a Processor block.
3. Use a Trainer block to predict `X` -> Y 
4. Test and score the pipeline.
