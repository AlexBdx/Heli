STRIDE=8
EPOCHS=200
NEG_RATIO=2

Doubled the amount of negatives, and this paid off. Now the model is at 90%.
To do:
- increase the (negative) class weight instead of bringing more pictures in
- 50 epochs seem enough to get convergence
- Pick a random test set? Might end up overfitting a specific test set but at random, could pick something very close to a training set and end up getting excellent, non representative results.
