#include "../src/tree/DecisionTreeClassifier.h"
#include "../src/_utils.h"
#include <iostream>
#include <string>

// the datatype of your dataset
using datatype = int;

int main(){
    /*
    * Sex: {0, 1}
    * Pclass: {1, 2, 3}
    * Embarked: {0, 1, 2}
    * Survived (binary target): {0, 1}
    */
    std::string file_path{"./titanic.csv"};
    algo::util::DataFrame dataframe{file_path, ',', true};
    auto df = dataframe.df<datatype>();
    algo::DecisionTreeClassifier<datatype> cls{5, 0};
    cls.fit(df);
    cls.PrintTree();
    std::cout << "##### Inference #######" << std::endl;
    std::vector<std::vector<datatype>> new_observations{
        {1, 3, 1}, // 0
        {0, 1, 2}, // 1
        {1, 0, 2}, // 0
        {1, 1, 1}  // 0
    };
    auto pred = cls.predict(new_observations);
    for (auto &c: pred)
        std::cout << c << std::endl;    
    return 0;
 };