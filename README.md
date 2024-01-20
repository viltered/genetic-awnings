# genetic-awnings
A genetic algorithm demo born out frustration with a heatwave. Optimizes the shape of canopies (awnings) to keep sunlight out during the summer, but not during the winter.

The [sunposition](https://github.com/s-bear/sun-position) module is used to calculate the direction of sunlight throughout the year. Each generation, some random days are picked and the canopies' fitness is evaluated, and the worst ones are replaced with offspring from the best.

A simple mesh is used to represent the canopies. The z coordinate is optimized while the x and y coordinates are fixed. As this is a poor representation of all possible shapes, the resulting canopy is jagged. But it achieves its goal none the less.

