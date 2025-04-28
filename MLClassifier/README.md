# ML Classifiers to detect assembly architectures

Success on 4th model iteration, with ~2000 samples.

## Problem

Given a binary blob (presumably excluding header data), classify it to one of 12 architectures. The architectures in question are
avr, alphaev56, arm, m68k, mips, mipsel, powerpc, s390, sh4, sparc, x86_64, and xtensa. Some of these are fairly distinct architectures,
some are more ambigous. 

## My Approach

I built out a classifier in python building off the base api calls. My first approach revolved around training off each api call to collect
labeled data. That was eventually changed to a system of training based on accuracy and confidence to ensure minority classes like mips was 
porportional to the rest of the data. I used a random forest classifier, assuming this would hit the perfect blend between speed and decision making.

The factors I looked at are as follows:

Byte Frequency: This is an easy discrete thing to look at, different opcodes and assemblies will have different skews of bytes
Entropy: Also easy to look at, may vary from architecture to architecture
Bigram Frequency: When looking at sequences of 2 bytes, see what stands out may hint at an architecture
Mean/Variance/Kurtosis/SKew: Why not?
Bitvector: See which values are always present, may hint at what CPU instruction set
OpCoes: Never fully implemented but certian opcodes are more distinctive than others and may indicate an assembly
Endianness: Because MIPS vs MIPSEL was a time


Ultimately I wanted to extract data on the behavior, structure, semantics, and variations in hopes of linking those traits
together to identify the architecture. 

### Models

Random Forest - As a baseline I decided to stand up a simple Random Forest model. This was iterated on several times to tune
learnign and training

XGBoost

## Notes

I HATE MIPS VS MIPSEL I HATE MIPS VS MIPSEL I HATE MIPS VS MIPSEL


