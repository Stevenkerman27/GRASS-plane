**Dependencies**
grass_pytorch should be run with Python 3.x and PyTorch 2.9.

grass_pytorch depends on torchfold which is a pytorch tool developed by [Illia Polosukhin](https://github.com/ilblackdragon). It is used for dynamic batching the computations in a dynamic computation graph. The computations across all nodes of all trees are batched based on their module names and dispatched to GPU for parallelization. 
