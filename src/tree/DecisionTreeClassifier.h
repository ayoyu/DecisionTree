#pragma once
#include <vector>
#include <set>
#include <unordered_map>
#include <iostream>
#include <tuple>
#include <memory>
#include "../_utils.h"


namespace algo{

class Constraint: public I_Printable{
    public:
        Constraint() = default;
        Constraint(size_t idx_f, int l)
            :index_feature(idx_f), limit(l)
        {};
        int get_limit() const;
        size_t get_index_feature() const;
        bool yes_record(const int& categorical_elem) const;
        bool true_false_record(const std::vector<int>& obs) const;
        virtual void print(std::ostream& os) const override;
        virtual ~Constraint(){};
    private:
        size_t index_feature;
        int limit;
};

class Node: public I_Printable{
    public:
        Node() = default;
        Node(double& g, size_t& l, size_t& s)
            :gini(g), level(l), nbr_samples(s)
        {};
        int get_class_value() const;
        Constraint get_constrain() const;
        void add_left_child(std::unique_ptr<Node>&& left);
        void add_right_child(std::unique_ptr<Node>&& right);
        virtual void print(std::ostream& os) const override;
        virtual ~Node(){};
    private:
        Constraint constraint;
        double gini;
        std::unique_ptr<Node> left_child;
        std::unique_ptr<Node> right_child;
        size_t level{0};
        size_t nbr_samples;
        int class_value{NULL};
        friend class DecisionTreeClassifier;
};

std::ostream& operator<<(std::ostream& os, const I_Printable& obj){
    obj.print(os);
    return os; 
};
std::vector<Constraint> feature_constraints(size_t& index_feature, std::set<int>& feature_values);
std::unordered_map<size_t, std::vector<Constraint>> records_constraints(const std::vector<std::vector<int>>& records);

struct RecordsSpliter{
    std::vector<std::vector<int>> left_records;
    std::vector<int> left_classes;
    std::vector<std::vector<int>> right_records;
    std::vector<int> right_classes;
};

RecordsSpliter split(const Constraint& ct, const std::vector<std::vector<int>>& records); 

struct Spliter{
    Constraint c;
    double left_gini;
    double right_gini;
    std::vector<std::vector<int>> left_records;
    std::vector<std::vector<int>> right_records;
    size_t left_nbr_samples;
    size_t right_nbr_samples;
};

double sub_gini_index(std::vector<int>& classes);

Spliter best_split(const std::vector<std::vector<int>>& records);

class DecisionTreeClassifier{
    public:
        DecisionTreeClassifier(size_t min_num, int default_class, size_t max_depth)
            :_min_num(min_num), _default_class(default_class), _max_depth(max_depth)
        {};
        DecisionTreeClassifier(size_t min_num, int default_class)
            :_min_num(min_num),  _default_class(default_class)
        {};
        void fit(std::vector<std::vector<int>>& records);
    private:
        void _BuildTree(std::vector<std::vector<int>>& records, std::unique_ptr<Node>& node, size_t depth);
    private:
        size_t _min_num;
        int _default_class;
        size_t _max_depth;
        std::unique_ptr<Node> _root = std::make_unique<Node>();
};

}