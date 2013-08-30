#include "test.h"

#include <vector>

std::vector<int *> get_numbers(int n)
{
    std::vector<int *> v;
    for(int i = 0; i < n; ++i)
    {
        int *val = new int;
        *val = i;
        v.push_back(val);
    }
    return v;
}


void set_numbers(std::vector<int *> *v)
{
    for(int i = 0; i < v->size(); ++i)
    {
        delete (*v)[i];
    }
	delete v;
}
