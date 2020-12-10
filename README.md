# PropaGAT
Experimenting with Graph Attentional Layers for detecting propaganda

For data visit https://propaganda.qcri.org/semeval2020-task11/

This project aims to detect political propaganda in news articles using graph attention networks (GATs). 
The implementation uses two GAT networks:
a) One to encode sentences as fully connected, directed graphs with words as nodes and,
b) One to encode articles as fully connected, directed graphs with sentences as nodes.

The goal is to not only detect propaganda in article level, but also use attention maps to detect propaganda in sentence level or possibly fragment/phrase level for improved explainability.

PS. The documentation and time efficiency and command line interactions are horrible for now. Bear with me please.
