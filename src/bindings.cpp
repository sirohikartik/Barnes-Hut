#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "simulation.hpp"

namespace py = pybind11;

PYBIND11_MODULE(black_hole_core, m) {
    m.doc() = "Barnes-Hut Octree N-body simulation with relativistic/pseudo-Newtonian Black Hole";

    py::class_<BlackHoleSimulation>(m, "Simulation")
        .def(py::init<double, double, double, double>(),
             py::arg("G") = 1.0,
             py::arg("c") = 60.0,
             py::arg("theta") = 0.6,
             py::arg("softening") = 0.08)
        
        .def("set_black_hole", [](BlackHoleSimulation& sim, double mass,
                                  double x, double y, double z,
                                  double vx, double vy, double vz) {
            sim.set_black_hole(mass, {x, y, z}, {vx, vy, vz});
        }, py::arg("mass"), py::arg("x") = 0.0, py::arg("y") = 0.0, py::arg("z") = 0.0,
           py::arg("vx") = 0.0, py::arg("vy") = 0.0, py::arg("vz") = 0.0)

        .def("add_body", [](BlackHoleSimulation& sim, double mass,
                            double x, double y, double z,
                            double vx, double vy, double vz) {
            return sim.add_body(mass, {x, y, z}, {vx, vy, vz});
        }, py::arg("mass"), py::arg("x"), py::arg("y"), py::arg("z"),
           py::arg("vx") = 0.0, py::arg("vy") = 0.0, py::arg("vz") = 0.0)

        .def("init_accretion_disk", &BlackHoleSimulation::init_accretion_disk,
             py::arg("n"), py::arg("r_min"), py::arg("r_max"), py::arg("total_disk_mass"),
             py::arg("thickness_ratio") = 0.03, py::arg("eccentricity") = 0.0, py::arg("seed") = 42)

        .def("init_star_cluster", [](BlackHoleSimulation& sim, int n,
                                     double cx, double cy, double cz,
                                     double cvx, double cvy, double cvz,
                                     double radius, double total_cluster_mass, unsigned int seed) {
            sim.init_star_cluster(n, {cx, cy, cz}, {cvx, cvy, cvz}, radius, total_cluster_mass, seed);
        }, py::arg("n"), py::arg("cx"), py::arg("cy"), py::arg("cz"),
           py::arg("cvx"), py::arg("cvy"), py::arg("cvz"),
           py::arg("radius"), py::arg("total_cluster_mass"), py::arg("seed") = 123)

        .def("step", &BlackHoleSimulation::step, py::arg("dt") = 0.01)
        .def("steps", &BlackHoleSimulation::steps, py::arg("num_steps"), py::arg("dt") = 0.01)
        .def("clear_bodies", &BlackHoleSimulation::clear_bodies)

        .def("get_positions", [](const BlackHoleSimulation& sim) {
            size_t n = sim.bodies.size();
            py::array_t<double> arr({n, (size_t)3});
            auto r = arr.mutable_unchecked<2>();
            for (size_t i = 0; i < n; ++i) {
                r(i, 0) = sim.bodies[i].pos.x;
                r(i, 1) = sim.bodies[i].pos.y;
                r(i, 2) = sim.bodies[i].pos.z;
            }
            return arr;
        })

        .def("get_velocities", [](const BlackHoleSimulation& sim) {
            size_t n = sim.bodies.size();
            py::array_t<double> arr({n, (size_t)3});
            auto r = arr.mutable_unchecked<2>();
            for (size_t i = 0; i < n; ++i) {
                r(i, 0) = sim.bodies[i].vel.x;
                r(i, 1) = sim.bodies[i].vel.y;
                r(i, 2) = sim.bodies[i].vel.z;
            }
            return arr;
        })

        .def("get_masses", [](const BlackHoleSimulation& sim) {
            size_t n = sim.bodies.size();
            py::array_t<double> arr(n);
            auto r = arr.mutable_unchecked<1>();
            for (size_t i = 0; i < n; ++i) {
                r(i) = sim.bodies[i].mass;
            }
            return arr;
        })

        .def("get_active", [](const BlackHoleSimulation& sim) {
            size_t n = sim.bodies.size();
            py::array_t<bool> arr(n);
            auto r = arr.mutable_unchecked<1>();
            for (size_t i = 0; i < n; ++i) {
                r(i) = sim.bodies[i].active;
            }
            return arr;
        })

        .def("get_black_hole_pos", [](const BlackHoleSimulation& sim) {
            py::array_t<double> arr(3);
            auto r = arr.mutable_unchecked<1>();
            r(0) = sim.black_hole.pos.x;
            r(1) = sim.black_hole.pos.y;
            r(2) = sim.black_hole.pos.z;
            return arr;
        })

        .def("get_black_hole_vel", [](const BlackHoleSimulation& sim) {
            py::array_t<double> arr(3);
            auto r = arr.mutable_unchecked<1>();
            r(0) = sim.black_hole.vel.x;
            r(1) = sim.black_hole.vel.y;
            r(2) = sim.black_hole.vel.z;
            return arr;
        })

        .def_property_readonly("black_hole_mass", [](const BlackHoleSimulation& sim) { return sim.black_hole.mass; })
        .def_property_readonly("rs", [](const BlackHoleSimulation& sim) { return sim.rs; })
        .def_property_readonly("swallowed_count", [](const BlackHoleSimulation& sim) { return sim.swallowed_count; })
        .def_property_readonly("total_accreted_mass", [](const BlackHoleSimulation& sim) { return sim.total_accreted_mass; })
        .def_property_readonly("active_count", &BlackHoleSimulation::active_body_count)
        .def_property_readonly("total_count", [](const BlackHoleSimulation& sim) { return static_cast<int>(sim.bodies.size()); })
        .def_property_readonly("time", [](const BlackHoleSimulation& sim) { return sim.current_time; })
        .def_property_readonly("step_count", [](const BlackHoleSimulation& sim) { return sim.step_count; })
        .def_property_readonly("kinetic_energy", &BlackHoleSimulation::get_kinetic_energy)
        .def_property_readonly("octree_node_count", [](const BlackHoleSimulation& sim) { return static_cast<int>(sim.octree.nodes.size()); })

        .def_readwrite("theta", &BlackHoleSimulation::theta)
        .def_readwrite("softening", &BlackHoleSimulation::softening)
        .def_readwrite("use_paczynski_wiita", &BlackHoleSimulation::use_paczynski_wiita)
        .def_readwrite("fixed_black_hole", &BlackHoleSimulation::fixed_black_hole)

        // Export octree boxes as numpy array (K, 5): [cx, cy, cz, half_width, depth]
        .def("get_octree_boxes", [](const BlackHoleSimulation& sim, int max_boxes) {
            auto boxes = sim.octree.get_bounding_boxes(max_boxes);
            size_t k = boxes.size();
            py::array_t<double> arr({k, (size_t)5});
            auto r = arr.mutable_unchecked<2>();
            for (size_t i = 0; i < k; ++i) {
                r(i, 0) = boxes[i].center.x;
                r(i, 1) = boxes[i].center.y;
                r(i, 2) = boxes[i].center.z;
                r(i, 3) = boxes[i].half_width;
                r(i, 4) = static_cast<double>(boxes[i].depth);
            }
            return arr;
        }, py::arg("max_boxes") = 500);
}
