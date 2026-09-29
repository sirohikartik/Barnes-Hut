#pragma once

#include "vec3.hpp"

struct Body {
    int id{0};
    Vec3 pos;
    Vec3 vel;
    Vec3 acc;
    double mass{1.0};
    bool is_black_hole{false};
    bool active{true};

    Body() = default;
    Body(int id_, const Vec3& p, const Vec3& v, double m, bool is_bh = false)
        : id(id_), pos(p), vel(v), acc(0.0, 0.0, 0.0), mass(m), is_black_hole(is_bh), active(true) {}
};
