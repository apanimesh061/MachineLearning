# -------------------------------------------------------------------------------
# Name:        gmm
# Purpose:     Gaussian Mixture Model
#
# Author:      Animesh Pandey
#
# Created:     17/10/2015
# Copyright:   (c) Animesh Pandey 2015
# -------------------------------------------------------------------------------

from __future__ import division
import numpy as np
from tabulate import tabulate
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class Data(object):
    data = None
    columns = 0
    rows = 0

    def __init__(self, path='2gaussian.txt'):
        self.data = np.loadtxt(path, delimiter=' ', dtype='float32')
        self.rows, self.columns = self.data.shape

    def plot(self):
        if self.columns == 2:
            X = self.data[:, 0]
            Y = self.data[:, 1]
            plt.scatter(X, Y)
            plt.grid()
            plt.show()
        else:
            print "Cannot plot for dimensions >= 2"


class Gaussian(object):
    Z = None
    mu = None
    covariance_matrix = None
    prior = None
    data = None
    no_of_instances = 0

    def __init__(self, dimensions, param, no_of_instances, full_data):
        mean, covariance, total_points = param["mu"], param["covariance_matrix"], param["total_points"]
        self.prior = total_points / no_of_instances
        self.mu = mean
        self.Z = np.random.uniform(0, 1, no_of_instances)
        self.covariance_matrix = covariance
        self.dimensions = dimensions
        self.no_of_instances = no_of_instances
        self.data = full_data

    def e_step(self):
        covariance_matrix_det = np.linalg.det(self.covariance_matrix)
        covariance_matrix_inverse = np.linalg.inv(self.covariance_matrix)
        for (index, point) in enumerate(self.data):
            multiplier = ((2 * np.pi) ** (-self.dimensions / 2)) * (covariance_matrix_det ** (-0.5))
            exponent = np.exp(
                    -(0.5 * np.sum(np.matrix(np.subtract(point, self.mu)) * covariance_matrix_inverse * np.matrix(
                            np.subtract(point, self.mu)).transpose())))
            expected_prob = multiplier * exponent
            self.Z[index] = self.prior * expected_prob

    def m_step(self):
        new_covariance_matrix = np.zeros((self.dimensions, self.dimensions))
        new_mean = np.zeros(self.dimensions)
        for index, point in enumerate(self.data):
            cm_update = self.Z[index] * np.matrix(np.subtract(point, self.mu)).transpose() \
                        * np.matrix(np.subtract(point, self.mu))

            new_covariance_matrix += cm_update
            mu_update = self.Z[index] * point
            new_mean += mu_update
        norm_factor = np.sum(self.Z)
        self.covariance_matrix = new_covariance_matrix / norm_factor
        self.mu = new_mean / norm_factor
        self.prior = norm_factor / self.no_of_instances

    def __str__(self):
        output = "\nCovariance Matrix: {0}\nMean: {1}\nPrior: {2}\nApprox no. of points: {3}\n" \
            .format(self.covariance_matrix, self.mu, self.prior, int(self.prior * self.no_of_instances))
        return output


class Mixture(object):
    no_of_instances = 0
    dimensions = 0
    data = None
    no_of_distributions = 0

    def __init__(self, total_points, dimensions, no_of_distributions, full_data):
        self.no_of_instances = total_points
        self.dimensions = dimensions
        self.distributions = []
        self.data = full_data
        self.no_of_distributions = no_of_distributions

    def initalize_params(self):
        params = []
        splits = [int(self.no_of_instances / self.no_of_distributions)] * self.no_of_distributions
        for i in range(self.no_of_distributions):
            param = dict()
            param["mu"] = np.random.uniform(1, 10, self.dimensions)
            param["covariance_matrix"] = np.eye(self.dimensions)
            param["total_points"] = splits[i]
            self.distributions.append(
                    Gaussian(self.dimensions, param,
                             self.no_of_instances, self.data))

    def soft_cluster(self, max_iter):
        iteration = 0
        while True:
            for distribution in self.distributions:
                distribution.e_step()
            for i in xrange(self.no_of_instances):
                total_z = 0.0
                for distribution in self.distributions:
                    total_z += distribution.Z[i]
                for distribution in self.distributions:
                    distribution.Z[i] /= total_z
            for distribution in self.distributions:
                distribution.m_step()

            iteration += 1
            if iteration % 10 == 0:
                print "{0} iterations complete...".format(iteration)
            if iteration == max_iter:
                for distribution in self.distributions:
                    print distribution
                break


if __name__ == "__main__":
    data = Data()
    gmm2 = Mixture(data.rows, data.columns, 2, data.data)
    gmm2.initalize_params()
    gmm2.soft_cluster(max_iter=100)

    # data = Data("3gaussian.txt")
    # gmm3 = Mixture(data.rows, data.columns, 3, data.data)
    # gmm3.initalize_params()
    # gmm3.soft_cluster(max_iter=200)
