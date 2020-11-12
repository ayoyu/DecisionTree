#pragma once
#include <string>
#include <exception>

namespace algo
{
class FileNotFound: public std::exception{
public:
    FileNotFound(std::string msg)
        :_msg(msg)
    {
    }
    virtual const char* what() const noexcept{
        return _msg.c_str();
    }
private:
    std::string _msg;
};

class ValueError: public std::exception{
    public:
        ValueError(std::string msg)
            :_msg(msg)
        {};
        virtual const char* what() const noexcept{
            return _msg.c_str();
        }
    private:
        std::string _msg;
};

}
