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


namespace algo
{

// Read DataFrame from a file
class DataFrame{
public:
    DataFrame(std::string& fp, char d, bool h)
        :file_path(fp), delimiter(d), header(h)
        {};
    
    std::vector<std::vector<int>> df(){
        // TODO: include string and float datatype for columns
        std::ifstream in_file{file_path, std::ios::in};
        if (!in_file)
            throw FileNotFound("File not foud: " + file_path);
        std::vector<std::vector<int>> records;
        std::string row;
        if (header){
            std::string line_header;
            std::getline(in_file, line_header);
            columns = _split_header(line_header);
        }
        while(std::getline(in_file, row)){
            records.push_back(_split_records(row));
        }
        in_file.close();
        return records;
    }
    std::vector<std::string> get_columns() const {
        return columns;
    }

    
private:
    std::vector<int> _split_records(const std::string& line){
        std::string token;
        std::vector<int> tokens;
        std::istringstream lineStream(line);
        while(std::getline(lineStream, token, delimiter)){
           tokens.push_back(std::stoi(token));
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

class I_Printable{
    friend std::ostream& operator<<(std::ostream& os, const I_Printable& obj);
    public:
        virtual void print(std::ostream& os) const = 0;
        virtual ~I_Printable(){};
};
}
