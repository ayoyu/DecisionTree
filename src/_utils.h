#pragma once
#include <string>
#include <fstream>
#include <vector>
#include "Exceptions.h"
#include <tuple>
#include <sstream>
#include <type_traits>

namespace algo
{

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

}
