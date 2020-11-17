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
#include "_Tree.h"

namespace algo{

/* For DecisionTreeClassifier:
*       - labels are classes (int)
*       - Splitting criteria is gini_index
*/ 

template<typename T>
_tree::RecordsSpliter<T, int> split(const _tree::Constraint<T>& ct, const std::vector<std::vector<T>>& records){
    std::vector<std::vector<T>> left_records;
    std::vector<std::vector<T>> right_records;
    std::vector<int> left_classes;
    std::vector<int> right_classes;
    size_t index_feature = ct.get_index_feature();
    for (const auto &row: records){
       if (ct.yes_record(row.at(index_feature))){
           left_records.push_back(row);
           left_classes.push_back(row.back());
       }
       else{
           right_records.push_back(row);
           right_classes.push_back(row.back());
       }
    }
    _tree::RecordsSpliter<T, int> sp{
        left_records,
        left_classes,
        right_records,
        right_classes
    };
    return sp;

} 

template<typename T>
_tree::Spliter<T> best_split(const std::vector<std::vector<T>>& records){
    size_t size_records = records.size();
    std::unordered_map<size_t, std::vector<_tree::Constraint<T>>> constraints = _tree::records_constraints(records);
    double best_gini = 1;
    _tree::Constraint<T> best_c;
    double best_left_gini, best_right_gini;
    std::vector<std::vector<T>> best_left_records, best_right_records;
    for (const auto &fc: constraints){
        // testing every evaluated constraint to take the best one based on gini
        for (const auto &c : fc.second){
            _tree::RecordsSpliter<T, int> records_spliter = split(c, records);
            double left_gini = util::sub_gini_index(records_spliter.left_labels); // labels here are classes(int)
            double right_gini = util::sub_gini_index(records_spliter.right_labels); // labels here are classes (int)
            double gini = (1. * records_spliter.left_records.size() / size_records) * left_gini + (1. * records_spliter.right_records.size() / size_records) * right_gini; 
            if (gini < best_gini){
                best_gini = gini;
                best_c = c;
                best_left_gini = left_gini;
                best_right_gini = right_gini;
                best_left_records = records_spliter.left_records;
                best_right_records = records_spliter.right_records;
            }
        }
    }
    _tree::Spliter<T> sp{
        best_c,
        best_left_gini,
        best_right_gini,
        best_left_records,
        best_right_records,
        best_left_records.size(),
        best_right_records.size()

    };
    return sp;
}

template<typename T>
class DecisionTreeClassifier{
public:
    DecisionTreeClassifier(size_t min_num, int default_class, size_t max_depth)
        :_min_num(min_num), _default_class(default_class), _max_depth(max_depth)
    {};
    DecisionTreeClassifier(size_t min_num, int default_class)
        :_min_num(min_num),  _default_class(default_class)
    {};
    void fit(const std::vector<std::vector<T>>& records){
        // setting nbr_samples and gini for the root for printing purpose
        _root->nbr_samples = records.size();
        std::vector<int> init_classes;
        for (const auto &row: records)
            init_classes.push_back(row.back());
        _root->splitting_criteria = util::sub_gini_index(init_classes);
        // setting the number of features for controlling purpose when we do prediction
        nbr_of_features = records[0].size() - 1; 
        // build the tree
        _BuildTree(records, _root, 1);
    }
    void PrintTree() const{
        // BFS print mode
        std::queue<const std::unique_ptr<_tree::Node<T, int>> *> nodes_queue;
        nodes_queue.push(&(_root));
        while (!nodes_queue.empty()){
            auto node = nodes_queue.front();
            if ((*node)->left_child)
                nodes_queue.push(&((*node)->left_child));
            if ((*node)->right_child)
                nodes_queue.push(&((*node)->right_child));
            nodes_queue.pop();
            std::cout << **node << std::endl;
        }
    }
    std::vector<int> predict(const std::vector<std::vector<T>>& observations){
        std::vector<int> class_predictions;
        for (const auto &row: observations){
            if (row.size() != nbr_of_features)
                throw exceptions::ValueError("Number of features of the model must match the input. Model nbr of features is " + std::to_string(nbr_of_features));
            int pred = _Inference(row, _root);
            class_predictions.push_back(pred);
        }
        return class_predictions;
    }
private:
    void _BuildTree(const std::vector<std::vector<T>>& records, std::unique_ptr<_tree::Node<T, int>>& node, size_t depth){
        std::vector<int> classes;
        for (const auto& row: records)
            classes.push_back(row.back());
        // check if all records have the same class
        std::unordered_set<int> set_classes{classes.begin(), classes.end()};
        if (set_classes.size() == 1){
            std::unordered_set<int>::iterator it = set_classes.begin();
            node->class_value = *it;
            return;
        }
        // records less than min_num => set the default class to this leafNode
        if (records.size() < _min_num){
            node->class_value = _default_class;
            return;
        }
        // stop if depth exceeds max_depth (if max_depth is provided by the user)
        if (_max_depth != 0){
            if (depth >= _max_depth){
                node->class_value = _default_class;
                return;
            }
        }
        
        _tree::Spliter<T> sp = best_split(records);
        // No split left => take the majority class
        if (sp.left_records.empty() || sp.right_records.empty()){
            std::unordered_map<int, int> count = util::Counter(classes);
            int majority_class = util::most_common(count).first;
            node->class_value = majority_class;
            return;
        }
        node->constraint = sp.c;
        /********Left branch***************/
        std::unique_ptr<_tree::Node<T, int>> left_child = std::make_unique<_tree::Node<T, int>>(sp.left_splitting_criteria, depth, sp.left_nbr_samples);
        node->add_left_child(std::move(left_child));
        _BuildTree(sp.left_records, node->left_child, depth+1);
        /********Right branch***************/
        std::unique_ptr<_tree::Node<T, int>> right_child = std::make_unique<_tree::Node<T, int>>(sp.right_splitting_criteria, depth, sp.right_nbr_samples);
        node->add_right_child(std::move(right_child));
        _BuildTree(sp.right_records, node->right_child, depth+1);
    }
    int _Inference(const std::vector<T>& obs, std::unique_ptr<_tree::Node<T, int>>& node){
        if (node->is_leaf()){
            return node->class_value;
        }
        if (node->constraint.true_false_record(obs)){
            return _Inference(obs, node->left_child);
        }
        else{
            return _Inference(obs, node->right_child);
        }
    }
private:
    size_t _min_num;
    int _default_class;
    size_t _max_depth{0};
    std::unique_ptr<_tree::Node<T, int>> _root = std::make_unique<_tree::Node<T, int>>();
    size_t nbr_of_features;
};

}