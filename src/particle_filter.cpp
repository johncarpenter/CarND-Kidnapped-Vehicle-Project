/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>

#include "particle_filter.h"
#include "helper_functions.h"

using namespace std;

const vector<LandmarkObs> nearestNeighbours(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {

	vector<LandmarkObs> min_obs;

	 for (int i =0; i < observations.size(); i++){

		 LandmarkObs obs = observations[i];

		 int min_dist = INFINITY;
		 LandmarkObs min;

		 for (int j =0; j < predicted.size(); j++){
			 LandmarkObs pred = predicted[j];
			 double dis = dist(obs.x,obs.y,pred.x,pred.y);
			 if(dis < min_dist ){
				 min_dist = dis;
				 min = pred;
			 }

		 }

		 min_obs.push_back(min);

	 }

	 return min_obs;
}


void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of
	//   x, y, theta and their uncertainties from GPS) and all weights to 1.
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	num_particles = 300;

	default_random_engine gen;

	// Creates a normal (Gaussian) distribution for x,y,theta
	normal_distribution<double> dist_x(x, std[0]);
	normal_distribution<double> dist_y(y, std[1]);
	normal_distribution<double> dist_psi(theta, std[2]);


	for (int i = 0; i < num_particles; ++i) {

	  Particle particle = Particle();

		particle.id = i;
		particle.weight = 1;

		// Sample from  normal distrubtions
		particle.x = dist_x(gen);
		particle.y = dist_y(gen);
		particle.theta = dist_psi(gen);

		particles.push_back(particle);

	}

	weights.resize(num_particles, 1.0f);

	is_initialized = true;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

  default_random_engine gen;

	normal_distribution<double> dist_x(0, std_pos[0]);
	normal_distribution<double> dist_y(0, std_pos[1]);
	normal_distribution<double> dist_psi(0, std_pos[2]);

	for (int i = 0 ; i < particles.size(); i++){
		Particle particle = particles[i];

		double thetaf,xf,yf;

		if(abs(yaw_rate) > 0.0001){
			thetaf = particle.theta + yaw_rate*delta_t;
			xf = particle.x + velocity/yaw_rate * (sin (thetaf) - sin (particle.theta));
			yf = particle.y + velocity/yaw_rate * (cos(particle.theta) - cos(thetaf));
		}else{
			thetaf = 0;
			xf = particle.x + velocity*delta_t*cos(yaw_rate);
			yf = particle.y + velocity*delta_t*sin(yaw_rate);
		}

		thetaf += dist_psi(gen);
		xf += dist_x(gen);
		yf += dist_y(gen);

		particles[i].x = xf;
		particles[i].y = yf;
		particles[i].theta = thetaf;

	}

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
	//   implement this method and use it as a helper during the updateWeights phase.

 vector<LandmarkObs> min_obs;

	for (int i =0; i < observations.size(); i++){

		LandmarkObs obs = observations[i];

		int min_dist = INFINITY;
		LandmarkObs min;

		for (int j =0; j < predicted.size(); j++){
			LandmarkObs pred = predicted[j];
			double dis = dist(obs.x,obs.y,pred.x,pred.y);
			if(dis < min_dist ){
				min_dist = dis;
				min = pred;
			}

		}

		min_obs.push_back(min);

	}

	observations = min_obs;

}




void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation
	//   3.33. Note that you'll need to switch the minus sign in that equation to a plus to account
	//   for the fact that the map's y-axis actually points downwards.)
	//   http://planning.cs.uiuc.edu/node99.html

	for (int i = 0 ; i < particles.size(); i++){
		Particle particle = particles[i];

		// Filter out landmarks out of sensor range
		vector<LandmarkObs> lm_in_range;
		for(int j=0; j<map_landmarks.landmark_list.size(); j++){
			Map::single_landmark_s lm = map_landmarks.landmark_list[j];

			double dis = dist(particle.x,particle.y,lm.x_f,lm.y_f);
			if(dis <= sensor_range){
				LandmarkObs obs;
				obs.x = lm.x_f;
				obs.y = lm.y_f;
				obs.id = lm.id_i;
				lm_in_range.push_back(obs);
			}
		}

		// Transform the coordinates from the observations into the current particle domain
		vector<LandmarkObs> obs_trans;
		for(int j=0; j < observations.size();j++){
			LandmarkObs obs = observations[j];
			LandmarkObs lm_trans;
			lm_trans.x = particle.x + obs.x * cos(particle.theta) - obs.y * sin(particle.theta);
			lm_trans.y = particle.y + obs.x * sin(particle.theta) + obs.y * cos(particle.theta);
			lm_trans.id = obs.id;
			obs_trans.push_back(lm_trans);
		}

		// Find the nearest-neighbors for the transformed operations
		vector<LandmarkObs> obs_nn = nearestNeighbours(lm_in_range, obs_trans);

		// Calculate the weights from total probability
		double prob = 1.0f;
		for(int j=0; j < obs_trans.size(); j++){
			LandmarkObs obs = obs_trans[j];
			LandmarkObs obs_n = obs_nn[j];

			long double p = 1/(2*M_PI*std_landmark[0]*std_landmark[1])*exp(-(pow(obs.x-obs_n.x,2.0)/(2*pow(std_landmark[0],2.0))+pow(obs.y-obs_n.y,2.0)/(2*pow(std_landmark[1],2.0))));
 			prob *= p;
		}

		particles[i].weight = prob;
		weights[i] = prob;

	}

}

void ParticleFilter::resample() {
	default_random_engine gen;
	// TODO: Resample particles with replacement with probability proportional to their weight.
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	discrete_distribution<int> distribution(weights.begin(), weights.end());
	vector<Particle> particle_samples;

	for(unsigned i = 0; i < num_particles; i++)
	{
	    particle_samples.push_back(particles[distribution(gen)]);
	}
	particles = particle_samples;
}

void ParticleFilter::write(std::string filename) {
	// You don't need to modify this file.
	std::ofstream dataFile;
	dataFile.open(filename, std::ios::app);
	for (int i = 0; i < num_particles; ++i) {
		dataFile << particles[i].x << " " << particles[i].y << " " << particles[i].theta << "\n";
	}
	dataFile.close();
}
