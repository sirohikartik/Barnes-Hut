#pragma once

#include "vec3.hpp"
#include "body.hpp"
#include <vector>
#include <cmath>
#include <algorithm>
#include <array>

struct OctreeNode {
    Vec3 center;
    double half_width{0.0};
    double mass{0.0};
    Vec3 com{0.0, 0.0, 0.0};
    int body_idx{-1};       // >= 0 if leaf holding body
    std::array<int, 8> children;
    bool is_leaf{true};
    int count{0};
    int depth{0};

    OctreeNode() {
        children.fill(-1);
    }

    OctreeNode(const Vec3& c, double hw, int d = 0)
        : center(c), half_width(hw), mass(0.0), com(0.0, 0.0, 0.0),
          body_idx(-1), is_leaf(true), count(0), depth(d) {
        children.fill(-1);
    }
};

struct BoundingBox {
    Vec3 center;
    double half_width;
    int depth;
    int count;
    double mass;
};

class Octree {
public:
    std::vector<OctreeNode> nodes;
    double theta{0.6};
    double softening{0.05};
    double G{1.0};
    int max_depth{32};

    Octree() {
        nodes.reserve(4096);
    }

    void clear() {
        nodes.clear();
    }

    int allocate_node(const Vec3& center, double half_width, int depth) {
        nodes.emplace_back(center, half_width, depth);
        return static_cast<int>(nodes.size() - 1);
    }

    static inline int get_octant(const Vec3& pos, const Vec3& center) {
        int oct = 0;
        if (pos.x >= center.x) oct |= 1;
        if (pos.y >= center.y) oct |= 2;
        if (pos.z >= center.z) oct |= 4;
        return oct;
    }

    static inline Vec3 get_child_center(const Vec3& center, double quarter_width, int octant) {
        return {
            center.x + ((octant & 1) ? quarter_width : -quarter_width),
            center.y + ((octant & 2) ? quarter_width : -quarter_width),
            center.z + ((octant & 4) ? quarter_width : -quarter_width)
        };
    }

    void build(const std::vector<Body>& bodies) {
        clear();
        if (bodies.empty()) return;

        // Compute bounding box for active bodies
        Vec3 min_pos(1e30, 1e30, 1e30);
        Vec3 max_pos(-1e30, -1e30, -1e30);
        int active_count = 0;

        for (const auto& b : bodies) {
            if (!b.active) continue;
            active_count++;
            min_pos.x = std::min(min_pos.x, b.pos.x);
            min_pos.y = std::min(min_pos.y, b.pos.y);
            min_pos.z = std::min(min_pos.z, b.pos.z);
            max_pos.x = std::max(max_pos.x, b.pos.x);
            max_pos.y = std::max(max_pos.y, b.pos.y);
            max_pos.z = std::max(max_pos.z, b.pos.z);
        }

        if (active_count == 0) return;

        Vec3 center = (min_pos + max_pos) * 0.5;
        double max_span = std::max({
            max_pos.x - min_pos.x,
            max_pos.y - min_pos.y,
            max_pos.z - min_pos.z
        });
        // Extra margin to ensure points strictly inside
        double half_width = (max_span * 0.5) * 1.05 + 1.0;

        allocate_node(center, half_width, 0);

        for (size_t i = 0; i < bodies.size(); ++i) {
            if (bodies[i].active) {
                insert(0, static_cast<int>(i), bodies);
            }
        }
    }

    void insert(int node_idx, int body_id, const std::vector<Body>& bodies) {
        const Body& body = bodies[body_id];

        // We use an iterative approach to prevent deep recursion
        int current_idx = node_idx;

        while (true) {
            OctreeNode& node = nodes[current_idx];

            if (node.count == 0) {
                // Empty node -> store body
                node.body_idx = body_id;
                node.mass = body.mass;
                node.com = body.pos;
                node.count = 1;
                node.is_leaf = true;
                return;
            }

            if (node.is_leaf) {
                // Leaf with 1 body already
                int old_body_id = node.body_idx;

                // If identical body, do nothing
                if (old_body_id == body_id) return;

                const Body& old_body = bodies[old_body_id];

                // If max depth reached or bodies at exact same position, just accumulate mass
                if (node.depth >= max_depth || (old_body.pos - body.pos).norm_sq() < 1e-18) {
                    double total_mass = node.mass + body.mass;
                    if (total_mass > 0.0) {
                        node.com = (node.com * node.mass + body.pos * body.mass) / total_mass;
                    }
                    node.mass = total_mass;
                    node.count++;
                    return;
                }

                // Subdivide this leaf node
                node.is_leaf = false;
                node.body_idx = -1;

                double quarter = node.half_width * 0.5;
                int old_oct = get_octant(old_body.pos, node.center);
                int old_child = allocate_node(get_child_center(node.center, quarter, old_oct), quarter, node.depth + 1);
                nodes[current_idx].children[old_oct] = old_child;

                // Insert old body into child
                nodes[old_child].body_idx = old_body_id;
                nodes[old_child].mass = old_body.mass;
                nodes[old_child].com = old_body.pos;
                nodes[old_child].count = 1;
                nodes[old_child].is_leaf = true;

                // Update current node COM & mass
                double total_mass = nodes[current_idx].mass + body.mass;
                nodes[current_idx].com = (nodes[current_idx].com * nodes[current_idx].mass + body.pos * body.mass) / total_mass;
                nodes[current_idx].mass = total_mass;
                nodes[current_idx].count++;

                // Now route the new body to its octant
                int new_oct = get_octant(body.pos, nodes[current_idx].center);
                if (nodes[current_idx].children[new_oct] == -1) {
                    int new_child = allocate_node(get_child_center(nodes[current_idx].center, quarter, new_oct), quarter, nodes[current_idx].depth + 1);
                    nodes[current_idx].children[new_oct] = new_child;
                    nodes[new_child].body_idx = body_id;
                    nodes[new_child].mass = body.mass;
                    nodes[new_child].com = body.pos;
                    nodes[new_child].count = 1;
                    nodes[new_child].is_leaf = true;
                    return;
                } else {
                    current_idx = nodes[current_idx].children[new_oct];
                    continue;
                }
            } else {
                // Internal node: update mass and COM
                double total_mass = node.mass + body.mass;
                node.com = (node.com * node.mass + body.pos * body.mass) / total_mass;
                node.mass = total_mass;
                node.count++;

                int oct = get_octant(body.pos, node.center);
                if (node.children[oct] == -1) {
                    double quarter = node.half_width * 0.5;
                    int new_child = allocate_node(get_child_center(node.center, quarter, oct), quarter, node.depth + 1);
                    nodes[current_idx].children[oct] = new_child;
                    nodes[new_child].body_idx = body_id;
                    nodes[new_child].mass = body.mass;
                    nodes[new_child].com = body.pos;
                    nodes[new_child].count = 1;
                    nodes[new_child].is_leaf = true;
                    return;
                } else {
                    current_idx = node.children[oct];
                    continue;
                }
            }
        }
    }

    Vec3 compute_force_on(int body_idx, const std::vector<Body>& bodies) const {
        if (nodes.empty()) return {0.0, 0.0, 0.0};

        const Body& target = bodies[body_idx];
        Vec3 total_acc(0.0, 0.0, 0.0);
        double eps_sq = softening * softening;

        // Fast stack for tree traversal
        int stack[256];
        int stack_ptr = 0;
        stack[stack_ptr++] = 0;

        while (stack_ptr > 0) {
            int curr = stack[--stack_ptr];
            const OctreeNode& node = nodes[curr];

            if (node.count == 0) continue;

            if (node.is_leaf) {
                if (node.body_idx != body_idx && node.body_idx >= 0) {
                    Vec3 r = node.com - target.pos;
                    double dist_sq = r.norm_sq() + eps_sq;
                    double dist = std::sqrt(dist_sq);
                    double inv_dist3 = 1.0 / (dist_sq * dist);
                    total_acc += r * (G * node.mass * inv_dist3);
                }
            } else {
                Vec3 r = node.com - target.pos;
                double dist_sq = r.norm_sq();
                double dist = std::sqrt(dist_sq);
                double s = 2.0 * node.half_width;

                // Barnes-Hut multipole acceptance criterion
                if (s < theta * dist) {
                    double softened_sq = dist_sq + eps_sq;
                    double soft_dist = std::sqrt(softened_sq);
                    double inv_dist3 = 1.0 / (softened_sq * soft_dist);
                    total_acc += r * (G * node.mass * inv_dist3);
                } else {
                    for (int c = 0; c < 8; ++c) {
                        int child_idx = node.children[c];
                        if (child_idx != -1 && stack_ptr < 255) {
                            stack[stack_ptr++] = child_idx;
                        }
                    }
                }
            }
        }

        return total_acc;
    }

    std::vector<BoundingBox> get_bounding_boxes(int max_boxes = 500) const {
        std::vector<BoundingBox> boxes;
        boxes.reserve(std::min<size_t>(nodes.size(), max_boxes));
        for (const auto& node : nodes) {
            if (node.count > 0 && boxes.size() < static_cast<size_t>(max_boxes)) {
                boxes.push_back({node.center, node.half_width, node.depth, node.count, node.mass});
            }
        }
        return boxes;
    }
};
