/**
 * 基础自动微分框架 (Basic Autograd Framework) - C++版本
 * 支持标量和向量的自动微分计算
 */
#pragma once
#include <iostream>
#include <vector>
#include <memory>
#include <functional>
#include <cmath>
#include <iomanip>
#include <ranges>
#include <unordered_set>
#include <fstream>
#include <map>
#include <cassert>

using string = std::string;
using idx_range = std::tuple<int,int,int>; // start, stop, step

class Variable;
class DataView;
using VarPtr = std::shared_ptr<class Variable>;
enum Nodetype {
    intermediate,
    parameter,
    input,
    reference,
};
struct Edge {
    VarPtr parent;
    VarPtr child;
    bool updated;
    bool pass_grad = true; // whether to pass gradient to parent
    Edge(VarPtr parent, VarPtr child, bool updated, bool pass_grad = true) : parent(parent), child(child), updated(updated), pass_grad(pass_grad) {}
};
#include "dataview.hpp"
#include "utils.hpp"
#include "variable.hpp"

Edge* add_link(VarPtr parent, VarPtr child, bool updated=false){

    auto e = new Edge(parent, child, updated);
    parent->add_child(e);
    child->add_parent(e);
    return e;
}

#include "operations.hpp"
#include "optimizer.hpp"
#include "graph.hpp"
#include "recurrent.hpp"



