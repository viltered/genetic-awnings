# genetic-awnings
A genetic algorithm demo born out of frustration with a heatwave. Optimizes the shape of canopies/awnings to keep sunlight out during the summer, but not during the winter.

The [sunposition](https://github.com/s-bear/sun-position) module is used to calculate the direction of sunlight throughout the year. Each generation some random days are picked, the canopies' fitness is evaluated, and the worst ones are replaced with mutated offspring from the best. A score function dictates the reward/punishment for shade depending on the day of the year.

The canopies are represented by simple meshes. The z coordinate of each point is optimized while the x and y coordinates are fixed. This simple method yields jagged shapes, but the goal is achieved: during the hottest summer months the window is almost constantly covered in shade, and vice-versa during the height of winter.

![shade](https://github.com/viltered/genetic-awnings/assets/10100093/53cc6507-99c6-4aaa-8a0f-5d21ea834c01)

The shadow moves around the window during winter...

![winter](https://github.com/viltered/genetic-awnings/assets/10100093/d83c5038-ae70-4266-a9bb-ed085983b32f)

...and covers it during summer.

![summer](https://github.com/viltered/genetic-awnings/assets/10100093/d3833995-ff02-4dd2-b34c-78d73f55259a)

Is this a good way of solving this problem? No. But i wanted to make a genetic algorithm...
