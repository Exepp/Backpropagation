#include <NeuralNetwork.hpp>
#include <iostream>

int main()
{
    DenseNN dnn;
    dnn.addLayer<SigmoidActivFn>(16, 2);
    dnn.addLayer<SigmoidActivFn>(16);
    dnn.addLayer(2);

    int steps = 1e6;
    for (int i = 0; i < steps; ++i) {

        Real_t x = blaze::rand<Real_t>() * 2 - 1;
        Real_t y = blaze::rand<Real_t>() * 2 - 1;
        Real_t xT = x * x + y * x;
        Real_t yT = 2 * x * y - y * y + x * x * x;
        dnn.learn({ x, y }, { xT, yT });
        if (i + 10 >= steps)
            std::cout << dnn({ x, y }) << xT << ' ' << yT << "\n\n";
    }
    return 0;
}