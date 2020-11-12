# :u6e80: AlgoFun
## Template Library for Tree and Graph Algorithms
### Under development :construction: (:bulb:Examples will be provided at each stage of the implementation of certain algorithms)

## Cmake Building
```shell
$ mkdir build && cd build
$ cmake ..
$ cmake --build build/
```

## Examples
### 1. Decision Tree Classifier
#### 1.1. Binary classification
##### **Examples/binary_classification_tree.cpp**
```C++
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
```
##### **Examples/run_binary_classification_tree.sh**
```shell
#!/bin/sh
g++ -Wall -o binary_class_exec binary_classification_tree.cpp -L../build -Wl,-rpath=../build -lAlgo
```

#### 1.2. Multiclass classification
##### **Examples/multiclass_classification_tree.cpp**
```C++
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
```
##### **Examples/run_multiclass_classification_tree.sh**
```shell
#!/bin/sh
g++ -Wall -o multi_class_exec multiclass_classification_tree.cpp -L../build -Wl,-rpath=../build -lAlgo
```

## TODO:
- **Cmake: switch from SHARED library to an INTERFACE library (header-only)**
- **Cmake: include installation of the library**