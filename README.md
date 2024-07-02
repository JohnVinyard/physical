# Physical Modelling Synthesis

This repository is a place for me to learn about physical modelling synthesis, and implement different approaches.  

## Spring Mesh

The first, and maybe most versatile approach is to model physical objects as 
[nodes with mass connected in a mesh of springs](physical.py).

## Digital Waveguide Synthesis

## Source/Excitation

Many reverbs work by convolving a room impulse response with a source signal.  You can also emulate the
resonating object changing shape by convolving the impulse with several different impulse response signals
and cross-fading between them over time.