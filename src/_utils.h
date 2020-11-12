#pragma once
#include <string>
#include <fstream>
#include <vector>
#include "Exceptions.h"
#include <tuple>
#include <sstream>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <bits/stdc++.h>
#include <typeinfo>
#include <cmath>

namespace algo
{

namespace util{

// Read DataFrame from a file
class DataFrame{
public:
    DataFrame(std::string& fp, char d, bool h)
        :file_path(fp), delimiter(d), header(h)
        {};
    template<typename T>
    std::vector<std::vector<T>> df(){
        // it accepts only numerical data (int, float/double)
        // and data need to be with the same type
        std::ifstream in_file{file_path, std::ios::in};
        if (!in_file)
            throw FileNotFound("File not foud: " + file_path);
        std::vector<std::vector<T>> records;
        std::string row;
        if (header){
            std::string line_header;
            std::getline(in_file, line_header);
            columns = _split_header(line_header);
        }
        while(std::getline(in_file, row)){
            records.push_back(_split_records<T>(row));
        }
        in_file.close();
        return records;
    }
    std::vector<std::string> get_columns() const {
        return columns;
    }

    
private:
    template<typename T>
    std::vector<T> _split_records(const std::string& line){
        std::string token;
        std::vector<T> tokens;
        std::istringstream lineStream(line);
        if (typeid(T) == typeid(int)){
            while(std::getline(lineStream, token, delimiter)){
                tokens.push_back(std::stoi(token));
            }
        }
        else if (typeid(T) == typeid(double)){
            while(std::getline(lineStream, token, delimiter)){
                tokens.push_back(std::stod(token));
            }
        }
        else if (typeid(T) == typeid(float)){
            while(std::getline(lineStream, token, delimiter)){
                tokens.push_back(std::stof(token));
            }
        }
        return tokens;
    }
    std::vector<std::string> _split_header(const std::string& line){
        std::string token;
        std::vector<std::string> tokens;
        std::istringstream lineStream(line);
        while(std::getline(lineStream, token, delimiter)){
           tokens.push_back(token);
        }
        return tokens;
    }
    std::string file_path;
    char delimiter;
    bool header;
    std::vector<std::string> columns{};
};

template<typename T>
std::unordered_map<T, int> Counter(std::vector<T> v){
    std::unordered_map<T, int> count_v;
    for (const auto &elem: v){
        if (count_v.count(elem)){
            count_v[elem] += 1;
        }
        else{
            count_v[elem] = 1;
        }
    }
    return count_v;
};

template<typename T>
std::pair<T, int> most_common(std::unordered_map<T, int> mp){
    std::vector<std::pair<T, int>> mp_v;
    for (const auto &elem: mp){
        mp_v.push_back(elem);
    }
    std::sort(
        mp_v.begin(),
        mp_v.end(),
        [](const std::pair<T, int>& p1, const std::pair<T, int>& p2) -> bool{
            return p1.second > p2.second;
        }
    );
    return mp_v[0];
};

double sub_gini_index(std::vector<int>& classes){
    size_t size = classes.size();
    std::unordered_map<int, int> count = Counter(classes);
    double sum_prob = 0.;
    for (const auto &c: count){
        sum_prob += std::pow((1. * c.second / size), 2);
    }
    double gini = 1. - sum_prob;
    return gini;
};

class I_Printable{
    friend std::ostream& operator<<(std::ostream& os, const I_Printable& obj);
    public:
        virtual void print(std::ostream& os) const = 0;
        virtual ~I_Printable(){};
};
}
}
