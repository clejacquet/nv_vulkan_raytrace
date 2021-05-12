#ifndef APPLICATION_HPP
#define APPLICATION_HPP

#include <memory>

class Application 
{
private:
    class Impl;
    std::unique_ptr<Impl> _impl;

public:
    Application();
    ~Application();

    void run();
};


#endif