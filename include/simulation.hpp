#pragma once

#include "vec3.hpp"
#include "body.hpp"
#include "octree.hpp"
#include <vector>
#include <cmath>
#include <random>
#include <iostream>
#include <string>
#include <map>

class BlackHoleSimulation {
public:
    double G{1.0};
    double c{60.0};              // Speed of light in simulation units
    double theta{0.6};           // Barnes-Hut opening angle
    double softening{0.08};      // Softening parameter for N-body collisions
    double current_time{0.0};
    int step_count{0};

    // Central Black Hole
    Body black_hole;
    double rs{0.0};              // Schwarzschild radius = 2 G M / c^2
    bool use_paczynski_wiita{true};
    bool fixed_black_hole{false}; // If true, BH does not drift from reactions
    int swallowed_count{0};
    double total_accreted_mass{0.0};

    // Other orbiting bodies (stars, disk particles)
    std::vector<Body> bodies;
    Octree octree;

    BlackHoleSimulation(double G_ = 1.0, double c_ = 60.0, double theta_ = 0.6, double softening_ = 0.08)
        : G(G_), c(c_), theta(theta_), softening(softening_) {
        octree.G = G;
        octree.theta = theta;
        octree.softening = softening;
        // Default central Black Hole
        set_black_hole(1000.0, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0});
    }

    void set_black_hole(double mass, const Vec3& pos = {0,0,0}, const Vec3& vel = {0,0,0}) {
        black_hole = Body(-1, pos, vel, mass, true);
        update_rs();
    }

    void update_rs() {
        rs = (2.0 * G * black_hole.mass) / (c * c);
    }

    int add_body(double mass, const Vec3& pos, const Vec3& vel) {
        int id = static_cast<int>(bodies.size());
        bodies.emplace_back(id, pos, vel, mass, false);
        return id;
    }

    void clear_bodies() {
        bodies.clear();
        swallowed_count = 0;
        total_accreted_mass = 0.0;
        current_time = 0.0;
        step_count = 0;
    }

    // Compute Black Hole gravitational acceleration on a given position
    Vec3 compute_bh_acceleration(const Vec3& pos) const {
        Vec3 r_vec = black_hole.pos - pos;
        double r = r_vec.norm();

        if (r <= rs) {
            // Inside horizon
            return {0.0, 0.0, 0.0};
        }

        if (use_paczynski_wiita) {
            // Paczyński-Wiita pseudo-Newtonian potential:
            // a = - G * M / (r - rs)^2 * (r_vec / r)
            double r_eff = r - rs;
            if (r_eff < 1e-4) r_eff = 1e-4;
            double inv_r = 1.0 / r;
            double a_mag = (G * black_hole.mass) / (r_eff * r_eff);
            return r_vec * (a_mag * inv_r);
        } else {
            // Standard Newtonian with softening
            double r_sq = r * r + softening * softening;
            double inv_r3 = 1.0 / (r_sq * std::sqrt(r_sq));
            return r_vec * (G * black_hole.mass * inv_r3);
        }
    }

    // Initialize an accretion disk with Keplerian / Paczyński-Wiita circular velocities
    void init_accretion_disk(int n, double r_min, double r_max, double total_disk_mass,
                            double thickness_ratio = 0.03, double eccentricity = 0.0,
                            unsigned int seed = 42) {
        std::mt19937_64 rng(seed);
        std::uniform_real_distribution<double> u_dist(0.0, 1.0);
        std::normal_distribution<double> z_dist(0.0, 1.0);
        std::normal_distribution<double> v_turb(0.0, 0.02);

        double star_mass = total_disk_mass / n;

        for (int i = 0; i < n; ++i) {
            // Surface density ~ 1/r distribution
            double u = u_dist(rng);
            double r = r_min + u * (r_max - r_min);
            double phi = u_dist(rng) * 2.0 * M_PI;

            // Disk vertical scale height
            double z = z_dist(rng) * (r * thickness_ratio);

            double x = r * std::cos(phi);
            double y = r * std::sin(phi);

            // Circular velocity in Paczyński-Wiita or Newtonian field
            double v_circ = 0.0;
            if (use_paczynski_wiita && r > rs) {
                // v_circ^2 / r = G * M / (r - rs)^2
                v_circ = std::sqrt((G * black_hole.mass * r) / ((r - rs) * (r - rs)));
            } else {
                v_circ = std::sqrt((G * black_hole.mass) / r);
            }

            // Slight eccentricity and turbulence
            v_circ *= (1.0 + eccentricity * std::sin(phi * 2.0));

            // Tangential velocity vector (-sin(phi), cos(phi), 0)
            double vx = -v_circ * std::sin(phi) + v_turb(rng) * v_circ;
            double vy =  v_circ * std::cos(phi) + v_turb(rng) * v_circ;
            double vz =  v_turb(rng) * v_circ * 0.2;

            add_body(star_mass, {x + black_hole.pos.x, y + black_hole.pos.y, z + black_hole.pos.z},
                                {vx + black_hole.vel.x, vy + black_hole.vel.y, vz + black_hole.vel.z});
        }
    }

    // Initialize an infalling or orbiting star cluster
    void init_star_cluster(int n, const Vec3& cluster_center, const Vec3& cluster_vel,
                          double cluster_radius, double total_cluster_mass, unsigned int seed = 123) {
        std::mt19937_64 rng(seed);
        std::uniform_real_distribution<double> u_dist(0.0, 1.0);
        std::normal_distribution<double> norm_dist(0.0, 1.0);

        double star_mass = total_cluster_mass / n;

        for (int i = 0; i < n; ++i) {
            // Plummer distribution
            double u = u_dist(rng);
            double r = cluster_radius / std::sqrt(std::pow(u, -2.0 / 3.0) - 1.0);
            if (r > cluster_radius * 4.0) r = cluster_radius * 4.0;

            double theta_angle = std::acos(2.0 * u_dist(rng) - 1.0);
            double phi_angle = u_dist(rng) * 2.0 * M_PI;

            double x = r * std::sin(theta_angle) * std::cos(phi_angle);
            double y = r * std::sin(theta_angle) * std::sin(phi_angle);
            double z = r * std::cos(theta_angle);

            // Internal velocity dispersion
            double v_esc = std::sqrt(2.0 * G * total_cluster_mass / std::sqrt(r * r + cluster_radius * cluster_radius));
            double vx = norm_dist(rng) * 0.2 * v_esc;
            double vy = norm_dist(rng) * 0.2 * v_esc;
            double vz = norm_dist(rng) * 0.2 * v_esc;

            add_body(star_mass, cluster_center + Vec3(x, y, z), cluster_vel + Vec3(vx, vy, vz));
        }
    }

    // Check accretion / collision with event horizon
    void check_accretions() {
        // Event horizon boundary (particles within horizon are captured)
        double horizon_sq = rs * rs;

        Vec3 bh_momentum = black_hole.vel * black_hole.mass;

        for (auto& b : bodies) {
            if (!b.active) continue;

            Vec3 diff = b.pos - black_hole.pos;
            if (diff.norm_sq() <= horizon_sq) {
                // Swallowed by black hole!
                b.active = false;
                swallowed_count++;
                total_accreted_mass += b.mass;

                // Inelastic collision conservation of momentum and mass
                bh_momentum += b.vel * b.mass;
                black_hole.mass += b.mass;
            }
        }

        if (!fixed_black_hole && black_hole.mass > 0.0) {
            black_hole.vel = bh_momentum / black_hole.mass;
        }

        update_rs();
    }

    // Full acceleration calculation for all bodies
    void compute_all_accelerations() {
        // 1. Build Barnes-Hut Octree for all active bodies
        octree.G = G;
        octree.theta = theta;
        octree.softening = softening;
        octree.build(bodies);

        // 2. Compute accelerations
        Vec3 bh_reaction_acc(0.0, 0.0, 0.0);

        for (size_t i = 0; i < bodies.size(); ++i) {
            if (!bodies[i].active) {
                bodies[i].acc = {0.0, 0.0, 0.0};
                continue;
            }

            // Force from central Black Hole
            Vec3 a_bh = compute_bh_acceleration(bodies[i].pos);

            // Force from other bodies via Barnes-Hut Octree
            Vec3 a_oct = octree.compute_force_on(static_cast<int>(i), bodies);

            bodies[i].acc = a_bh + a_oct;

            // Reaction on black hole from this body (Newton's 3rd law)
            if (!fixed_black_hole) {
                Vec3 r_to_bh = bodies[i].pos - black_hole.pos;
                double dist_sq = r_to_bh.norm_sq() + softening * softening;
                double inv_dist3 = 1.0 / (dist_sq * std::sqrt(dist_sq));
                bh_reaction_acc += r_to_bh * (G * bodies[i].mass * inv_dist3);
            }
        }

        if (!fixed_black_hole) {
            black_hole.acc = bh_reaction_acc;
        } else {
            black_hole.acc = {0.0, 0.0, 0.0};
        }
    }

    // Velocity-Verlet Integration Step
    void step(double dt) {
        if (step_count == 0) {
            compute_all_accelerations();
        }

        double half_dt = 0.5 * dt;

        // 1. First half-step velocity kick and full position drift
        for (auto& b : bodies) {
            if (!b.active) continue;
            b.vel += b.acc * half_dt;
            b.pos += b.vel * dt;
        }

        if (!fixed_black_hole) {
            black_hole.vel += black_hole.acc * half_dt;
            black_hole.pos += black_hole.vel * dt;
        }

        // 2. Check event horizon captures
        check_accretions();

        // 3. Compute new accelerations at t + dt
        compute_all_accelerations();

        // 4. Second half-step velocity kick
        for (auto& b : bodies) {
            if (!b.active) continue;
            b.vel += b.acc * half_dt;
        }

        if (!fixed_black_hole) {
            black_hole.vel += black_hole.acc * half_dt;
        }

        current_time += dt;
        step_count++;
    }

    void steps(int num_steps, double dt) {
        for (int i = 0; i < num_steps; ++i) {
            step(dt);
        }
    }

    int active_body_count() const {
        int cnt = 0;
        for (const auto& b : bodies) {
            if (b.active) cnt++;
        }
        return cnt;
    }

    double get_kinetic_energy() const {
        double ke = 0.5 * black_hole.mass * black_hole.vel.norm_sq();
        for (const auto& b : bodies) {
            if (b.active) {
                ke += 0.5 * b.mass * b.vel.norm_sq();
            }
        }
        return ke;
    }
};
