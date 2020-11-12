#include "DecisionTreeClassifier.h"
#include "../_utils.h"
#include <iostream>
#include <set>
#include <cmath>
#include <unordered_set>
#include <queue>

namespace algo
{

int Constraint::get_limit() const{
    return limit;
} 
size_t Constraint::get_index_feature() const{
    return index_feature;
}
bool Constraint::yes_record(const int& categorical_elem) const{
    return categorical_elem < limit;
}
bool Constraint::true_false_record(const std::vector<int>& obs) const{
    return obs.at(index_feature) < limit;
}
void Constraint::print(std::ostream& os) const {
    os << "feature: " << index_feature << " || Q: " << "< " << limit << std::endl;
}
int Node::get_class_value() const {
    return class_value;
}
Constraint Node::get_constrain() const {
    return constraint;
}
void Node::add_left_child(std::unique_ptr<Node>&& left){
    left_child = std::move(left);
}
void Node::add_right_child(std::unique_ptr<Node>&& right){
    right_child = std::move(right);
}
void Node::print(std::ostream& os) const {
    os << "level: " << level << std::endl;
    os << "gini: " << gini << " || " << "samples: " << nbr_samples << std::endl;
    if  (class_value == 100)
        os << constraint;
    else
        // leaf nodes don't have constraints
        os << "class: " << class_value << std::endl;    
}
std::vector<Constraint> feature_constraints(size_t& index_feature, std::set<int>& feature_values){
    // feature_values is a sorted set asc
    std::vector<Constraint> constraints;
    std::set<int>::reverse_iterator r_it;
    for (r_it=feature_values.rbegin(); r_it!=std::prev(feature_values.rend()); r_it++){
        constraints.push_back(Constraint(index_feature, *r_it));
    }
    return constraints;
};

std::unordered_map<size_t, std::vector<Constraint>> records_constraints(const std::vector<std::vector<int>>& records){
    std::unordered_map<size_t, std::vector<Constraint>> constraints;
    size_t nbr_features = records.at(0).size() - 1;
    for (size_t i=0; i<nbr_features; i++){
        std::set<int> unique_v{};
        for (const auto &row: records){
            unique_v.insert(row[i]);
        }
        std::vector<Constraint> fc = feature_constraints(i, unique_v);
        if (!fc.empty()){
            constraints[i] = fc;
        } 
    }
    return constraints;
};

RecordsSpliter split(const Constraint& ct, const std::vector<std::vector<int>>& records){
    std::vector<std::vector<int>> left_records;
    std::vector<std::vector<int>> right_records;
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
    RecordsSpliter sp{
        left_records,
        left_classes,
        right_records,
        right_classes
    };
    return sp;
};

double sub_gini_idex(const std::vector<int>& classes){
    size_t size = classes.size();
    std::unordered_map<int, int> count = Counter(classes);
    double sum_prob = 0.;
    for (const auto &c: count){
        sum_prob += std::pow((1. * c.second / size), 2);
    }
    double gini = 1. - sum_prob;
    return gini;

};
Spliter best_split(const std::vector<std::vector<int>>& records){
    size_t size_records = records.size();
    std::unordered_map<size_t, std::vector<Constraint>> constraints = records_constraints(records);
    double best_gini = 1;
    Constraint best_c;
    double best_left_gini, best_right_gini;
    std::vector<std::vector<int>> best_left_records, best_right_records;
    for (const auto &fc: constraints){
        // testing every evaluated constraint to take the best one based on gini
        for (const auto &c : fc.second){
            RecordsSpliter records_spliter = split(c, records);
            double left_gini = sub_gini_idex(records_spliter.left_classes);
            double right_gini = sub_gini_idex(records_spliter.right_classes);
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
    Spliter sp{
        best_c,
        best_left_gini,
        best_right_gini,
        best_left_records,
        best_right_records,
        best_left_records.size(),
        best_right_records.size()

    };
    return sp;

};

void DecisionTreeClassifier::_BuildTree(const std::vector<std::vector<int>>& records, std::unique_ptr<Node>& node, size_t depth){
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
    
    Spliter sp = best_split(records);
    // No split left => take the majority class
    if (sp.left_records.empty() || sp.right_records.empty()){
        std::unordered_map<int, int> count = Counter(classes);
        int majority_class = most_common(count).first;
        //node->set_class_value(majority_class);
        node->class_value = majority_class;
        return;
    }
    node->constraint = sp.c;
    /********Left branch***************/
    std::unique_ptr<Node> left_child = std::make_unique<Node>(sp.left_gini, depth, sp.left_nbr_samples);
    node->add_left_child(std::move(left_child));
    _BuildTree(sp.left_records, node->left_child, depth+1);
    /********Right branch***************/
    std::unique_ptr<Node> right_child = std::make_unique<Node>(sp.right_gini, depth, sp.right_nbr_samples);
    node->add_right_child(std::move(right_child));
    _BuildTree(sp.right_records, node->right_child, depth+1);

};

void DecisionTreeClassifier::fit(const std::vector<std::vector<int>>& records){
    // setting nbr_samples and gini for the root for printing purpose
    _root->nbr_samples = records.size();
    std::vector<int> init_classes;
    for (const auto &row: records)
        init_classes.push_back(row.back());
    _root->gini = sub_gini_idex(init_classes);
    // setting the number of features for controlling purpose when we do prediction
    nbr_of_features = records[0].size() - 1; 
    // build the tree
    _BuildTree(records, _root, 1);
};
void DecisionTreeClassifier::PrintTree() const {
    // BFS print mode
    std::queue<const std::unique_ptr<Node> *> nodes_queue;
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
};
int DecisionTreeClassifier::_Inference(const std::vector<int>& obs, std::unique_ptr<Node>& node){
    // stop condition for leaf node (only a leaf node has a valid class_value)
    if (node->class_value != 100){
        return node->class_value;
    }
    if (node->constraint.true_false_record(obs)){
        return _Inference(obs, node->left_child);
    }
    else{
        return _Inference(obs, node->right_child);
    }
};
std::vector<int> DecisionTreeClassifier::predict(const std::vector<std::vector<int>>& observations){
    std::vector<int> class_predictions;
    for (const auto &row: observations){
        if (row.size() != nbr_of_features)
            throw ValueError("Number of features of the model must match the input. Model nbr of features is " + std::to_string(nbr_of_features));
        int pred = _Inference(row, _root);
        class_predictions.push_back(pred);
    }
    return class_predictions;
};
} // namespace algo

