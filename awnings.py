

from typing import Iterable
import datetime as dt
import multiprocessing

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sunposition

class Canopy:
    def __init__(self, grid):
        self.grid = grid
    
    @classmethod
    def from_random(cls, x_min, x_max, y_min, y_max, x_res, y_res, height_min, height_max):
        # build a random sunscreen mesh, ready for optimization
        # number of points in each direction = number of tiles +1
        x_res += 1
        y_res += 1
        grid = np.meshgrid(np.linspace(x_min, x_max, x_res), np.linspace(y_min, y_max, y_res))
        grid = np.array(grid)
        grid = np.append(grid, np.random.rand(1, y_res, x_res) * (height_max - height_min) + height_min, axis=0)
        return cls(grid)

    def get_kinky(self, kinkiness=0.1, height_only=True):
        # procreation - create a mutated version of this sunscreen
        
        if height_only:
            mutation = (np.random.rand(*self.grid.shape) - 0.5) * kinkiness
            # only z axis is allowed to mutate
            mutation[:-1] = 0.
            return Canopy(self.grid + mutation)
            # todo:  ...
        else:
            mutation = (np.random.rand(*self.grid.shape) - 0.5) * kinkiness
            return Canopy(self.grid + mutation)
        

    
    def plot(self, ax):
        ax.plot_surface(self.grid[0], self.grid[1], self.grid[2], cmap='coolwarm')


class VerticalWindow:
    """
    A normal non-slanted window, with position on earth, orientation, and dimensions.
    """
    def __init__(self, window_azimuth, y_min, y_max, z_min, z_max, lat, lon, alt):
        self.window_azimuth = window_azimuth  # azimuth = 0 corresponds to a window facing north (?)
        self.y_min, self.y_max = y_min, y_max
        self.z_min, self.z_max = z_min, z_max
        self.lat = lat
        self.lon = lon
        self.alt = alt
        self.build_year_profile()
        self.refresh_evaluation_criteria()
        
    def calculate_shadow(self, canopy: Canopy, sun_azimuth, sun_zenith, ax=None, plot_samples=True):
        """Estimates the fraction of the window which is covered in the given canopy's shade for the given sun position.

        Args:
            sunscreen (SunScreen): the sunscreen to calculate the shadow coverage of
            sun_azimuth (np.ndarray): the azimuth of the sun relative to the azimuth of the window's normal
            sun_zenith (np.ndarray): zenith of the sun
            ax (plt.Axes): matplotlib axes object to plot the shadow coverage
        Returns:
            float: fraction of the window that is in the shade
        """
        
        # check if sunlight comes from behind the window, ensuring 100% shadow
        assert np.all(sun_azimuth < np.pi) and np.all(sun_zenith < np.pi / 2.), 'The sun is not facing the window at the given orientation.'
        if not isinstance(sun_azimuth, Iterable):
            sun_azimuth = [sun_azimuth]
            sun_zenith = [sun_zenith]
        if not isinstance(sun_azimuth, np.ndarray):
            sun_azimuth = np.array(sun_azimuth)
            sun_zenith = np.array(sun_zenith)
        assert(len(sun_azimuth) == len(sun_zenith))
        
        # 1. project grid
        # project sunscreen coordinates onto window plane (yz-plane)
        shadow = np.tile(canopy.grid[1:,...,np.newaxis].copy(), len(sun_azimuth))
        expanded_x = np.tile(canopy.grid[0,...,np.newaxis], len(sun_azimuth))
        shadow[0] += expanded_x / np.tan(sun_azimuth)
        shadow[1] -= expanded_x / np.tan(sun_zenith)
        
        # 2. turn grid of points into mesh/list of triangles described by point coordinates
        t = np.array([
            np.concatenate((shadow[:,:-1,:-1], shadow[:,1:,1:]), axis=2),
            np.tile(shadow[:,1:,:-1], (1, 1, 2, 1)),
            np.tile(shadow[:,:-1,1:], (1, 1, 2, 1))
        ])
        
        # 3. check for N points if they are inside the mesh
        shaded = []
        
        denominator = (t[1,1] - t[2,1]) * (t[0,0] - t[2,0]) + (t[2,0] - t[1,0]) * (t[0,1] - t[2,1])
        for y_, z_ in zip(self.y, self.z):
            alpha = ((t[1,1] - t[2,1]) * (y_ - t[2,0]) + (t[2,0] - t[1,0]) * (z_ - t[2,1])) / denominator
            beta = ((t[2,1] - t[0,1]) * (y_ - t[2,0]) + (t[0,0] - t[2,0]) * (z_ - t[2,1])) / denominator
            gamma = 1. - alpha - beta
            shaded.append(np.any((alpha > 0.) & (beta > 0.) & (gamma > 0.), axis=(0,1)))

        shaded = np.array(shaded)
        
        # if a matplotlib.Axes object is provided, plot the projection to it
        if ax is not None:
            assert shadow.shape[-1] == 1, 'Can not plot shadow of multiple sun angles at the same time.'
            shadow = shadow.reshape(shadow.shape[:-1])
            
            # plot vertices of projected sunscreen grid
            # ax.scatter(np.zeros_like(shadow[0]), shadow[0], shadow[1], color='black')
            
            # # plot shadow mesh (2/3 edges of each triangle)
            # for i in range(t.shape[-1]):
            #     ax.plot(np.zeros_like(t[:,0,i]), t[:,0,i], t[:,1,i], color='black')
            
            # plot grid without diagonal lines
            for i in range(shadow.shape[-1]):
                ax.plot(np.zeros_like(shadow[0,:,i]), shadow[0,:,i], shadow[1,:,i], color='black')
            for i in range(shadow.shape[-2]):
                ax.plot(np.zeros_like(shadow[0,i,:]), shadow[0,i,:], shadow[1,i,:], color='black')
            
            # plot sample points
            if plot_samples:
                shaded = shaded.reshape(shaded.shape[:-1])
                not_shaded = np.logical_not(shaded)
                ax.scatter(np.zeros_like(self.y[shaded]), self.y[shaded], self.z[shaded], color='grey')
                ax.scatter(np.zeros_like(self.y[not_shaded]), self.y[not_shaded], self.z[not_shaded], color='yellow')

        return np.sum(shaded, axis=0) / len(self.y)

    def plot(self, ax):
        y, z = np.meshgrid((self.y_min, self.y_max), (self.z_min, self.z_max))
        ax.plot_surface(np.zeros_like(y) + 1e-10, y, z, color='blue', alpha=0.3)

    def get_sun_position(self, date):
        return sunposition.sunpos(date, self.lat, self.lon, self.alt)
    
    def build_year_profile(self, samples=50):
        """ Pre-calculate information about the sun's position relative to this window. """
        
        # 50 time samples per day for the year 2024
        start = dt.datetime(2024, 1, 1)
        end = dt.datetime(2025, 1, 1)
        self.days = pd.date_range(start, end, 365 * samples + 1, tz='Europe/Brussels', inclusive='left')
        
        # convert time samples to unix timestamps
        self.days = self.days.astype(np.int64) / 1e9
        self.days = self.days.to_numpy().reshape((365, samples))
        
        # compute sun position for each time stamp
        self.sun_azimuth, self.sun_zenith, ra, dec, hour_angle = sunposition.sunpos(self.days, self.lat, self.lon, self.alt)
        
        # convert to radians and set azimuth angle relative to the window normal
        self.sun_zenith *= np.pi / 180
        self.sun_azimuth = (self.sun_azimuth / 180 * np.pi - self.window_azimuth + np.pi / 2.) % (2. * np.pi)

        # find time samples where the sun is visible from the window
        self.sun_mask = (self.sun_azimuth < np.pi) & (self.sun_zenith < np.pi / 2.)
        self.sunny_moments = np.array(np.where(self.sun_mask))
    
    def refresh_evaluation_criteria(self, num_times=40, num_samples=80):
        """ Select a new set of random times throughout the year and random sun rays to be used for evaluation from now on. """
        time_indices = np.random.choice(self.sunny_moments.shape[1], size=num_times, replace=False)
        self.times = self.sunny_moments[:,time_indices]
        self.y = np.random.rand(num_samples) * (self.y_max - self.y_min) + self.y_min
        self.z = np.random.rand(num_samples) * (self.z_max - self.z_min) + self.z_min

    def calculate_score(self, canopy, scoring_curve):
        """ Calculate score for a given canopy and scoring curve. Uses the last selected criteria. """
        daily_sun_weights = scoring_curve[self.times[0,:]]
        shadow = self.calculate_shadow(
            canopy,
            self.sun_azimuth[self.times[0],self.times[1]],
            self.sun_zenith[self.times[0],self.times[1]]
        )
        total_scoring_curve_weight = np.abs(daily_sun_weights).sum()
        sun_score = np.dot(shadow, daily_sun_weights)
        
        return sun_score / total_scoring_curve_weight
    
        # # add an additional term for the simplicity of the canopy
        # smoothness_score = 0.
        # horizontal = np.linalg.norm(canopy.grid[:,:-1,:] - canopy.grid[:,1:,:], axis=0)
        # vertical = np.linalg.norm(canopy.grid[:,:,:-1] - canopy.grid[:,:,1:], axis=0)
        # horizontal = (horizontal / np.average(horizontal) - np.ones_like(horizontal)) ** 2
        # vertical = (vertical / np.average(vertical) - np.ones_like(vertical)) ** 2
        
        # lengths = np.concatenate((horizontal.flatten(), vertical.flatten()))
        # smoothness_score = - np.average(lengths)
        # print(f'smoothness penalty: {smoothness_score}')
        
        # return sun_score / total_scoring_curve_weight + smoothness_score * 1.0

    def full_evaluation(self, canopy, scoring_curve, num_samples=80, ax=None):
        """ Evaluate the given canopy for all pre-calculated times, and the last selected light rays. """

        daily_sun_weights = scoring_curve[self.sunny_moments[0,:]]
        shadow = self.calculate_shadow(
            canopy,
            self.sun_azimuth[self.sunny_moments[0],self.sunny_moments[1]],
            self.sun_zenith[self.sunny_moments[0],self.sunny_moments[1]]
        )

        if ax is not None:
            # gather indices marking each day
            day_boundaries = np.concatenate((
                np.array((-1,), int),
                np.where(np.diff(self.sunny_moments[0]))[0],
                np.array((len(shadow),))
            ))
        
            # collect mean shadow for each day
            shadow_by_day = []
            for i, j in zip(day_boundaries[:-1], day_boundaries[1:]):
                shadow_by_day.append(np.mean(shadow[i:j+1]))
            shadow_by_day = np.array(shadow_by_day)
         
            # plot score function   
            ax2 = ax.twinx()
            ax2.plot(np.arange(365), scoring_curve, color='red', label='score function')
            ax2.set_xlabel('score')
            ax2.set_ylabel('score factor')
            ax2.legend()
            
            # plot mean window shadow coverage
            ax.plot(np.arange(365), shadow_by_day, color='blue', label='daily mean shadow coverage')
            ax.set_xlabel('day of the year')
            ax.set_ylabel('mean window shadow coverage')
            ax.legend()
            
        
        total_scoring_curve_weight = np.abs(daily_sun_weights).sum()
        sun_score = np.dot(shadow, daily_sun_weights)
        
        return sun_score / total_scoring_curve_weight



if __name__ == '__main__':
    
    GENERATIONS = 150
    POPULATION_SIZE = 40
    TEST_COORDS = (50.8541, 4.3508, 20)  # longitude, latitude, altitude (m) for Brussels, Belgium
    AZIMUTH = np.pi  # window orientation
    
    # define window
    window = VerticalWindow(AZIMUTH, -0.5, 0.5, 0., 1., *TEST_COORDS)
    
    # plot initial canopy shape, shadow, and light ray samples
    canopy = Canopy.from_random(0., 1., -1., 1., 4, 8, 1., 1.)
    fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
    window.plot(ax)
    canopy.plot(ax)
    window.calculate_shadow(canopy, np.pi / 4, np.pi / 4, ax=ax)
    plt.show()

    # set up population
    population = np.array([Canopy.from_random(0., 1., -1., 1., 4, 8, 1., 1.) for _ in range(POPULATION_SIZE)])
    
    # define fitness function for the genetic algorithm
    scoring_curve = -np.cos(np.linspace(0, 2 * np.pi, 365))
    def score_function(canopy):
        return window.calculate_score(canopy, scoring_curve)
    
    # perform optimization
    start = dt.datetime.now()
    
    for generation in range(GENERATIONS):
        # calculate population scores
        window.refresh_evaluation_criteria(num_times=200, num_samples=100)
        threadpool = multiprocessing.Pool(8)
        scores = np.array(threadpool.map(score_function, population))
        
        # sort population by score
        sorted_indices = np.argsort(scores)[::-1]
        scores = scores[sorted_indices]
        population = population[sorted_indices]
    
        # print top scores for this generation
        # ! note: each generation is evaluated on a different random set of days, some may be harder than others.
        #         therefore the score will fluctuate randomly as well, and short term regressions are possible.
        print(f'generation {generation} max scores:   1. {scores[0]:.5f}   2. {scores[1]:.5f}   3. {scores[2]:.5f}')

        # replace worst half of population with new offspring
        kinkiness = 0.02 + (1. - generation / GENERATIONS) * 0.25
        children = np.array([s.get_kinky(kinkiness=kinkiness) for s in population[:population.shape[0] // 2]])
        population[len(population) // 2:] = children

    end = dt.datetime.now()
    print(f'elapsed time: {end - start}')

    # plot the best scoring canopy of the last generation
    best = population[0]
    fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
    window.plot(ax)
    best.plot(ax)
    window.calculate_shadow(population[0], np.pi / 4, np.pi / 4, ax=ax)
    ax.axes.set_xlim(0, 2.0)
    ax.axes.set_ylim(-1.0, 1.0)
    ax.axes.set_zlim(0, 1.5)
    ax.azim = -30
    ax.elev = 15
    plt.show()

    # # plot shade fraction throughout the year
    # score = window.full_evaluation(best, scoring_curve, 80, ax=plt.gca())
    # fig.tight_layout()
    # plt.show()

    # # plot + save shade throughout a day in winter and summer
    # for season, day in zip(("winter, summer"), (0, 180)):
        
    #     moments = np.where(window.sunny_moments[0] == day)
    #     moments = window.sunny_moments[1][moments]
        
    #     for i, (az, ze) in enumerate(zip(window.sun_azimuth[day, moments], window.sun_zenith[day, moments])):
            
    #         fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
    #         fig.set_size_inches(12, 10)
    #         window.plot(ax)
    #         best.plot(ax)
    #         window.calculate_shadow(best, az, ze, ax, plot_samples=False)
    #         ax.axes.set_xlim(0, 2.0)
    #         ax.axes.set_ylim(-1.0, 1.0)
    #         ax.axes.set_zlim(0, 1.5)
    #         ax.azim = -30
    #         ax.elev = 10
    #         ax.axes.xaxis.set_ticklabels([])
    #         ax.axes.yaxis.set_ticklabels([])
    #         ax.axes.zaxis.set_ticklabels([])
            
    #         plt.tight_layout()
    #         plt.savefig(f'figures/{day}_{i:0>3}.png', bbox_inches='tight')  # use ffmpeg to convert to .gif
    #         # plt.show()
    #         print(f'day {day}, moment {i:0>3} saved')
        
