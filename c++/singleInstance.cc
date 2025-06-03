#include<bits/stdc++.h>

class Singleton
{

    public:
    Singleton & operator=(const Singleton&)=delete;
    Singleton(const Singleton&)=delete;
    Singleton(Singleton&&)=delete;
    Singleton& operator=(Singleton&&)=delete;
    // {
    //     std::cout<<"Copy constructor called."<<std::endl;

    // }
    static Singleton& get_instance()
    {
        static Singleton instance;
        // Singleton *instance=nullptr;

        return instance;
    }

    private:
    Singleton()
    {
        std::cout<<"Default constructor called."<<std::endl;
    }

        // constexpr static Singleton * instance=nullptr;
};

int main()
{

    Singleton& instance =Singleton::get_instance();

    std::cout<<"address of instance: "<<&instance<<std::endl;
    Singleton& instance2 = Singleton::get_instance();
    std::cout<<"address of instance2: "<<&instance2<<std::endl;
    std::cout << "Singleton instance created successfully." << std::endl;

    return 0;
}