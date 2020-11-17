#pragma once
#include <vector>
#include <unordered_map>
#include <tuple>
#include <memory>
#include "../_utils.h"
#include <iostream>
#include <set>
#include <cmath>
#include <unordered_set>
#include <queue>
#include <typeinfo>

namespace algo{

template<typename T>
class DecisionTreeClassifier;

template<typename T>
class DecisionTreeRegressor;

namespace _tree{


template<typename T>
class Constraint{
// constraint template: typename T -> features type
public:
    Constraint() = default;
    Constraint(size_t idx_f, T l)
        :index_feature(idx_f), limit(l)
    {};
    T get_limit() const {
        return limit;
    }
    size_t get_index_feature() const {
        return index_feature;
    }
    bool yes_record(const T& categorical_elem) const {
        return categorical_elem < limit;
    }
    bool true_false_record(const std::vector<T>& obs) const {
        return obs.at(index_feature) < limit;
    }
    void print(std::ostream& os) const {
        os << "feature: " << index_feature << " || Q: " << "< " << limit << std::endl;
    };
private:
    size_t index_feature;
    T limit;
};

template<typename T, typename L>
class Node{
// Node template: typename T -> features type; typename L -> labels type (classification || regression)
public:
    Node() = default;
    Node(double& criteria, size_t& l, size_t& s)
        :splitting_criteria(criteria), level(l), nbr_samples(s)
    {};
    L get_class_value() const {
        return class_value;
    }
    Constraint<T> get_constrain() const {
        return constraint;
    }
    void add_left_child(std::unique_ptr<Node<T, L>>&& left){
        left_child = std::move(left);
    }
    void add_right_child(std::unique_ptr<Node<T, L>>&& right){
        right_child = std::move(right);
    }
    void print(std::ostream& os) const {
        os << "level: " << level << std::endl;
        if (typeid(L) == typeid(int)){
            os << "gini: " << splitting_criteria << " || " << "samples: " << nbr_samples << std::endl;
        }
        else if (typeid(L) == typeid(double)){
            os << "variance: " << splitting_criteria << " || " << "samples: " << nbr_samples << std::endl;
        }
        if (!this->is_leaf())
            os << constraint;
        else
            // leaf nodes don't have constraints
            os << "class: " << class_value << std::endl;
    }
    bool is_leaf() const{
        return (!left_child && !right_child);
    }
private:
    Constraint<T> constraint;
    double splitting_criteria;
    std::unique_ptr<Node<T, L>> left_child;
    std::unique_ptr<Node<T, L>> right_child;
    size_t level{0};
    size_t nbr_samples;
    L class_value;
    friend class DecisionTreeClassifier<T>;
    friend class DecisionTreeRegressor<T>;
};

template<typename U>
std::ostream& operator<<(std::ostream& os, const U& obj){
    obj.print(os);
    return os; 
};
template<typename T>
std::vector<Constraint<T>> feature_constraints(size_t& index_feature, std::set<T>& feature_values){
    // feature_values is a sorted set asc
    std::vector<Constraint<T>> constraints;
    typename std::set<T>::reverse_iterator r_it;
    for (r_it=feature_values.rbegin(); r_it!=std::prev(feature_values.rend()); r_it++){
        constraints.push_back(Constraint<T>(index_feature, *r_it));
    }
    return constraints;
}

template<typename T>
std::unordered_map<size_t, std::vector<Constraint<T>>> records_constraints(const std::vector<std::vector<T>>& records){
    std::unordered_map<size_t, std::vector<Constraint<T>>> constraints;
    size_t nbr_features = records.at(0).size() - 1;
    for (size_t i=0; i<nbr_features; i++){
        std::set<T> unique_v{};
        for (const auto &row: records){
            unique_v.insert(row[i]);
        }
        std::vector<Constraint<T>> fc = feature_constraints(i, unique_v);
        if (!fc.empty()){
            constraints[i] = fc;
        } 
    }
    return constraints;
}

template<typename T, typename L>
struct RecordsSpliter{
    std::vector<std::vector<T>> left_records;
    std::vector<L> left_labels;
    std::vector<std::vector<T>> right_records;
    std::vector<L> right_labels;
};
 
template<typename T>
struct Spliter{
    Constraint<T> c;
    double left_splitting_criteria;
    double right_splitting_criteria;
    std::vector<std::vector<T>> left_records;
    std::vector<std::vector<T>> right_records;
    size_t left_nbr_samples;
    size_t right_nbr_samples;
};

}
}