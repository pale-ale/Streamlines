#pragma once

#include <vector>
#include <iostream> 

class FilterTarget{
    public:
    FilterTarget(int x, int y):targetX{x}, targetY{y}{
        data = std::vector<double>(x * y);
    };

    int targetX;
    int targetY;

    double Get(const int &x, const int &y) const{
        auto idx = y * targetX + x;
        if (0 > idx || idx >= targetX*targetY){
            // std::cout << "Bad Int Index: " << idx << " (y="<<y<<", x="<<x<<")" << " (max: " << x*y << ")\n";
        }
        return data[idx];
    }
    double Get(const double &x, const double &y) const{
        auto idx = ((int)(y * targetY)) * targetX + (int)(x * targetX);
        if (0 > idx || idx >= targetX*targetY){
            // std::cout << "Bad Double Index: " << idx << " (y="<<y<<", x="<<x<<")" << " (max: " << targetX*targetY << ")\n";
        }
        return data[idx];
    }
    void Coord(int idx, int &x, int &y){
        y = idx / targetX;
        x = idx % targetX;
    }

    std::vector<double> data;
};
