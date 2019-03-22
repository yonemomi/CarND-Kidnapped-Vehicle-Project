/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>
#include <random>
#include "helper_functions.h"

using std::normal_distribution;
using std::string;
using std::vector;

void ParticleFilter::init(double x, double y, double theta, double std[])
{
  /**
   * TODO: Set the number of particles. Initialize all particles to
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1.
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method
   *   (and others in this file).
   */
  num_particles = 1000; // TODO: Set the number of particles
  particles.clear();
  weights.clear();
  std::default_random_engine gen;
  normal_distribution<double> dist_x(x, std[0]);
  normal_distribution<double> dist_y(y, std[1]);
  normal_distribution<double> dist_theta(theta, std[2]);

  for (int i = 0; i < num_particles; ++i)
  {
    Particle particle;
    particle.id = i;
    particle.x = dist_x(gen);
    particle.y = dist_y(gen);
    particle.theta = dist_theta(gen);
    particle.weight = 1.0f;
    particles.push_back(particle);
    weights.push_back(1.0f);
  }
  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[],
                                double velocity, double yaw_rate)
{
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */
  std::default_random_engine gen;
  normal_distribution<double> dist_x(0, std_pos[0]);
  normal_distribution<double> dist_y(0, std_pos[1]);
  normal_distribution<double> dist_theta(0, std_pos[2]);
  // std::cout << "******"
  // << "prediction start" << std::endl;
  double x_f = 0;
  double y_f = 0;
  double theta_f = 0;
  for (int i = 0; i < num_particles; ++i)
  {
    // std::cout << "******"
    // << "particle_no" << i << std::endl;

    if (abs(yaw_rate) > 0.00001)
    {
      x_f = particles[i].x + (velocity / yaw_rate) * (sin(particles[i].theta + (yaw_rate * delta_t)) - sin(particles[i].theta));
      y_f = particles[i].y + (velocity / yaw_rate) * (cos(particles[i].theta) - cos(particles[i].theta + (yaw_rate * delta_t)));
      theta_f = particles[i].theta + (yaw_rate * delta_t);
    }
    else
    {
      x_f = particles[i].x + velocity * cos(particles[i].theta) * delta_t;
      y_f = particles[i].y + velocity * sin(particles[i].theta) * delta_t;
      theta_f = particles[i].theta;
    }
    particles[i].x = x_f + dist_x(gen);
    particles[i].y = y_f + dist_y(gen);
    particles[i].theta = theta_f + dist_theta(gen);
  }
  // std::cout << "******"
  // << "predictioni end" << std::endl;
}

double distance_2d(double dx, double dy)
{
  return sqrt(dx * dx + dy * dy);
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted,
                                     vector<LandmarkObs> &observations)
{
  /**
   * TODO: Find the predicted measurement that is closest to each
   *   observed measurement and assign the observed measurement to this
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will
   *   probably find it useful to implement this method and use it as a helper
   *   during the updateWeights phase.
   */
  for (int i = 0; i < observations.size(); ++i)
  {
    int minimum_observation_index = -1;
    double minimum_distance = std::numeric_limits<float>::max();
    for (int j = 0; j < predicted.size(); ++j)
    {
      double dx = predicted[j].x - observations[i].x;
      double dy = predicted[j].y - observations[i].y;
      double distance = distance_2d(dx, dy);
      if (distance < minimum_distance)
      {
        minimum_distance = distance;
        minimum_observation_index = j;
      }
    }
    observations[i].id = minimum_observation_index;
  }
}

vector<LandmarkObs> reduce_LandmarkObs_by_perspective(double sensor_range, Particle particle, Map map_landmarks)
{
  vector<LandmarkObs> predictions;
  for (int i = 0; i < map_landmarks.landmark_list.size(); ++i)
  {
    LandmarkObs landmark;
    landmark.id = map_landmarks.landmark_list[i].id_i;
    landmark.x = map_landmarks.landmark_list[i].x_f;
    landmark.y = map_landmarks.landmark_list[i].y_f;
    double distance = distance_2d(particle.x - landmark.x, particle.y - landmark.y);
    if (distance < sensor_range)
    {
      predictions.push_back(landmark);
    }
  }
  return predictions;
}

vector<LandmarkObs> transform_coordinate_vehicle_2_map(vector<LandmarkObs> observations, Particle particle)
{
  vector<LandmarkObs> transformed_observations;
  transformed_observations.clear();

  // std::cout << "******"
  // << "transform 1-1 num_observation" << observations.size() << std::endl;
  for (int i = 0; i < observations.size(); i++)
  {
    // std::cout << "******"
    // << "transform 1-2 id:" << observations[i].id << " x,y" << observations[i].x << "," << observations[i].y << std::endl;
    double x = cos(particle.theta) * observations[i].x - sin(particle.theta) * observations[i].y + particle.x;
    // std::cout << "******"
    // << "transform 1-3 id:" << observations[i].id << " x,y" << observations[i].x << "," << observations[i].y << std::endl;
    double y = sin(particle.theta) * observations[i].x + cos(particle.theta) * observations[i].y + particle.y;
    // std::cout << "******"
    // << "transform 1-4 " << std::endl;
    transformed_observations.push_back(LandmarkObs{observations[i].id, x, y});
    // std::cout << "******"
    // << "transform 1-5 " << std::endl;
  }
  return transformed_observations;
}

double calculate_particle_final_weight(vector<LandmarkObs> predictions, vector<LandmarkObs> observations, double std_landmark[])
{
  double weight = 1.0;

  for (int i = 0; i < observations.size(); ++i)
  {
    int obs_id = observations[i].id;
    double obs_x = observations[i].x;
    double obs_y = observations[i].y;
    double pred_x = predictions[obs_id].x;
    double pred_y = predictions[obs_id].y;
    double dx = pred_x - obs_x;
    double dy = pred_y - obs_y;
    /* Weight caluclation for specific particle*/
    double first_term = (2 * M_PI * std_landmark[0] * std_landmark[1]);
    double second_term = exp(-1 * (((dx * dx) / (2 * std_landmark[0] * std_landmark[0])) + ((dy * dy) / (2 * std_landmark[1] * std_landmark[1]))));
    weight *= second_term / first_term;
    // std::cout << "******"
    // << "caluculate_particle_final_weight:" << weight << "First: " << first_term << "Second: " << second_term << std::endl;
  }
  return weight;
}

void ParticleFilter::normalize_particle_weights()
{
  double sum_weight = 0.0;
  for (int i = 0; i < num_particles; ++i)
  {
    sum_weight += particles[i].weight;
  }
  for (int i = 0; i < num_particles; ++i)
  {
    particles[i].weight /= sum_weight;
    weights[i] = particles[i].weight;
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   const vector<LandmarkObs> &observations,
                                   const Map &map_landmarks)
{
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian
   *   distribution. You can read more about this distribution here:
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system.
   *   Your particles are located according to the MAP'S coordinate system.
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */
  // std::cout << "******"
  // << "update Weights" << std::endl;
  for (int i = 0; i < num_particles; ++i)
  {
    vector<LandmarkObs> predictions;
    vector<LandmarkObs> transformed_observations;
    predictions.clear();
    transformed_observations.clear();
    // std::cout << "******"
    // << "step1" << std::endl;
    predictions = reduce_LandmarkObs_by_perspective(sensor_range, particles[i], map_landmarks);
    // std::cout << "******"
    // << "step2" << std::endl;

    transformed_observations = transform_coordinate_vehicle_2_map(observations, particles[i]);
    // std::cout << "******"
    // << "step3" << std::endl;

    dataAssociation(predictions, transformed_observations);
    // std::cout << "******"
    // << "step4" << std::endl;

    particles[i].weight = calculate_particle_final_weight(predictions, transformed_observations, std_landmark);
  }
  normalize_particle_weights();
}

void ParticleFilter::resample()
{
  /**
   * TODO: Resample particles with replacement with probability proportional
   *   to their weight.
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */
  std::random_device rd;
  std::mt19937 gen(rd());
  vector<Particle> particles_new;
  std::discrete_distribution<> dist(weights.begin(), weights.end());
  for (int i = 0; i < num_particles; ++i)
  {
    int particle_idx = dist(gen);
    Particle resampled = particles[particle_idx];
    resampled.id = i;
    particles_new.push_back(resampled);
  }
  particles = particles_new;
}

void ParticleFilter::SetAssociations(Particle &particle,
                                     const vector<int> &associations,
                                     const vector<double> &sense_x,
                                     const vector<double> &sense_y)
{
  // particle: the particle to which assign each listed association,
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations = associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1); // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord)
{
  vector<double> v;

  if (coord == "X")
  {
    v = best.sense_x;
  }
  else
  {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1); // get rid of the trailing space
  return s;
}