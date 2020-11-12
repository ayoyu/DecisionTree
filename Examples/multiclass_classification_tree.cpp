#include "../src/tree/DecisionTreeClassifier.h"
#include "../src/_utils.h"
#include <iostream>
#include <string>

// the datatype of your dataset
using datatype = double;

int main(){
    /*
    * 1. sepal length in cm
    * 2. sepal width in cm
    * 3. petal length in cm
    * 4. petal width in cm
    * 5. species(multiclass target):
    -- Iris Setosa: 0
    -- Iris Versicolor: 1
    -- Iris Virginica: 2
    */
    std::string file_path{"./iris.csv"};
    algo::util::DataFrame dataframe{file_path, ',', true};
    auto df = dataframe.df<datatype>();
    algo::DecisionTreeClassifier<datatype> cls{5, 0};
    cls.fit(df);
    cls.PrintTree();
    std::cout << "##### Inference #######" << std::endl;
    std::vector<std::vector<datatype>> new_observations{
        {5.5,2.4,3.8,1.1},//1
        {4.9,2.5,4.5,1.7},//2
        {5.5,4.2,1.4,0.2},//0
        {6.4,3.2,4.5,1.5} //1
    };
    auto pred = cls.predict(new_observations);
    for (auto &c: pred)
        std::cout << c << std::endl;    
    return 0;
 };